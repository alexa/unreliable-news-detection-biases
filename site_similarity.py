import tensorflow as tf
from utils import load_jsonl, load_train_dev_test_json
import numpy as np
import os
import pickle as pkl
import tf_utils
from functools import partial
import pandas as pd

def maximum_mean_discrepancy(x, y, kernel=tf_utils.gaussian_kernel_matrix):
  r"""Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
  Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
  the distributions of x and y. Here we use the kernel two sample estimate
  using the empirical mean of the two distributions.
  MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
              = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },
  where K = <\phi(x), \phi(y)>,
    is the desired kernel function, in this case a radial basis kernel.
  Args:
      x: a tensor of shape [num_samples, num_features]
      y: a tensor of shape [num_samples, num_features]
      kernel: a function which computes the kernel in MMD. Defaults to the
              GaussianKernelMatrix.
  Returns:
      a scalar denoting the squared maximum mean discrepancy loss.
  """
  with tf.name_scope('MaximumMeanDiscrepancy'):
    # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
    cost = tf.reduce_mean(kernel(x, x))
    cost += tf.reduce_mean(kernel(y, y))
    cost -= 2 * tf.reduce_mean(kernel(x, y))

    # We do not allow the loss to become negative.
    cost = tf.where(cost > 0, cost, 0, name='value')
  return cost


def get_mmd_dist(source_samples, target_samples):
  """Adds a similarity loss term, the MMD between two representations.
  This Maximum Mean Discrepancy (MMD) loss is calculated with a number of
  different Gaussian kernels.
  Args:
    source_samples: a tensor of shape [num_samples, num_features].
    target_samples: a tensor of shape [num_samples, num_features].
    weight: the weight of the MMD loss.
    scope: optional name scope for summary tags.
  Returns:
    a scalar tensor representing the MMD loss value.
  """
  if type(source_samples) is list:
      source_samples = tf.convert_to_tensor(source_samples)
  if type(target_samples) is list:
      target_samples = tf.convert_to_tensor(target_samples)
  sigmas = [
      1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
      1e3, 1e4, 1e5, 1e6
  ]
  gaussian_kernel = partial(
      tf_utils.gaussian_kernel_matrix, sigmas=tf.constant(sigmas))

  loss_value = maximum_mean_discrepancy(
      source_samples, target_samples, kernel=gaussian_kernel)

  return loss_value


def get_coral_dist(source_samples, target_samples):
  """Adds a similarity loss term, the correlation between two representations.
  Args:
    source_samples: a tensor of shape [num_samples, num_features]
    target_samples: a tensor of shape [num_samples, num_features]
    weight: a scalar weight for the loss.
    scope: optional name scope for summary tags.
  Returns:
    a scalar tensor representing the correlation loss value.
  """
  if type(source_samples) is list:
      source_samples = tf.convert_to_tensor(source_samples)
  if type(target_samples) is list:
      target_samples = tf.convert_to_tensor(target_samples)
  with tf.name_scope('corr_loss'):
    source_samples -= tf.reduce_mean(source_samples, 0)
    target_samples -= tf.reduce_mean(target_samples, 0)

    source_samples = tf.nn.l2_normalize(source_samples, 1)
    target_samples = tf.nn.l2_normalize(target_samples, 1)

    source_cov = tf.matmul(tf.transpose(source_samples), source_samples)
    target_cov = tf.matmul(tf.transpose(target_samples), target_samples)

    corr_loss = tf.reduce_mean(tf.square(source_cov - target_cov))

  return corr_loss


def get_l2_dist(s_vecs, t_vecs):
    s_emb = tf.reduce_mean(s_vecs, axis=0)
    t_emb = tf.reduce_mean(t_vecs, axis=0)

    return tf.norm(s_emb-t_emb, ord='euclidean')


def get_cos_dist(s_vecs, t_vecs):
    s_emb = tf.reduce_mean(s_vecs, axis=0)
    t_emb = tf.reduce_mean(t_vecs, axis=0)
    cos_loss = tf.keras.losses.CosineSimilarity(axis=-1, reduction=tf.keras.losses.Reduction.NONE)
    cos_dist = cos_loss(s_emb, t_emb) + 1

    return cos_dist


def get_embeds(embed, data, embeds_type):
    if embeds_type=='tf':
        data = [e['title'] for e in data]
        return embed(data)
    elif embeds_type == 'allen':
        predictor = embed
        reader = predictor._dataset_reader
        data = [reader.text_to_instance({'title':e['title'], 'content': e['content'], 'label': '1'}) for e in data]

        results = []
        for inst in data:
            with predictor.capture_model_internals() as internals:
                predictor.predict_instance(inst)
            results.append(list(internals.values())[-4]['output'][0])


        return results
    else:
        raise NotImplementedError


def numerical_analysis(save_dir, site_pred_results, top_acc=True):
    with open(os.path.join(save_dir, 'site_data.pkl'), 'rb') as fr:
        site_data_dict = pkl.load(fr)
    with open(os.path.join(save_dir, 'site_emb.pkl'), 'rb') as fr:
        site_emb_dict = pkl.load(fr)
    with open(os.path.join(save_dir, 'sites.pkl'), 'rb') as fr:
        all_selected_sites = pkl.load(fr)
    sim_df_rr=pd.read_pickle(os.path.join(save_dir, 'sim_df_rr.pkl'))
    sim_df_ff=pd.read_pickle(os.path.join(save_dir, 'sim_df_ff.pkl'))
    sim_df_rf=pd.read_pickle(os.path.join(save_dir, 'sim_df_rf.pkl'))
    sim_df_fr=pd.read_pickle(os.path.join(save_dir, 'sim_df_fr.pkl'))

    r_names = set(sim_df_rr.columns.values.tolist())
    f_names = set(sim_df_ff.columns.values.tolist())

    site_results = load_jsonl(site_pred_results)[0]['source_results']

    all_keys = list(site_results.keys())

    for s in all_keys:
        if site_results[s]['size'] < 100:
            del site_results[s]

    sorted_results = sorted(site_results.items(), key=lambda x:x[1]['acc'], reverse=top_acc)

    top_same_scores = []
    top_rev_scores = []

    #print site name, site size, site accuracy and highest/ssmallest similarity w.r.t r/f training sites
    print(f"Site similarity results for {save_dir}")
    for s, s_dict in sorted_results[:10]:
        site_name = s

        if site_name in r_names:
            same_sim_sites = sim_df_rr.loc[:, site_name].astype(float)
            rev_sim_sites = sim_df_fr.loc[:, site_name].astype(float)
            top_same_site, top_same_score = same_sim_sites.idxmin(), same_sim_sites.min()
            top_rev_site, top_rev_score = rev_sim_sites.idxmin(), rev_sim_sites.min()
        elif site_name in f_names:
            same_sim_sites = sim_df_ff.loc[:, site_name].astype(float)
            rev_sim_sites = sim_df_rf.loc[:, site_name].astype(float)
            top_same_site, top_same_score = same_sim_sites.idxmin(), same_sim_sites.min()
            top_rev_site, top_rev_score = rev_sim_sites.idxmin(), rev_sim_sites.min()
        else:
            print(f"site {site_name} not found!!!")


        print(f"Site Name: {site_name}\t\tSite Size: {s_dict['size']}\t\tSite Acc: {s_dict['acc']}\t\tTop same site: {top_same_site}\t\tTop same score: {top_same_score}\t\tTop rev site: {top_rev_site}\t\tTop rev score: {top_rev_score}")

        top_same_scores.append(top_same_score)
        top_rev_scores.append(top_rev_score)

    return top_same_scores, top_rev_scores


def run_numerical_analysis():
    SAVE_DIRS = ["./output/site_sim_load/2label_domain_42_title",
                 "./output/site_sim_load/2label_domain_84_title",
                 "./output/site_sim_load/2label_domain_126_title",
                 "./output/site_sim_load/2label_domain_168_title",
                 "./output/site_sim_load/2label_domain_210_title"]
    SITE_PREDS = ['./output/rob_nela_2018_2label_tittxt_domain_42/source_dev_preds.json',
                  './output/rob_nela_2018_2label_tittxt_domain_84/source_dev_preds.json',
                  './output/rob_nela_2018_2label_tittxt_domain_126/source_dev_preds.json',
                  './output/rob_nela_2018_2label_tittxt_domain_168/source_dev_preds.json',
                  './output/rob_nela_2018_2label_tittxt_domain_210/source_dev_preds.json']
    save_dir_suffix = ["cos", "mmd", "coral", "l2"]
    for suf in save_dir_suffix:
        full_top_top_same_scores, full_top_top_rev_scores, full_bot_top_same_scores, full_bot_top_rev_scores = [], [], [], []
        for save_dir, site_pred in zip(SAVE_DIRS, SITE_PREDS):
                full_save_dir = save_dir + suf
                top_top_same_scores, top_top_rev_scores = numerical_analysis(save_dir=full_save_dir, site_pred_results=site_pred, top_acc=True)
                bot_top_same_scores, bot_top_rev_scores = numerical_analysis(save_dir=full_save_dir, site_pred_results=site_pred, top_acc=False)
                full_top_top_same_scores.extend(top_top_same_scores)
                full_bot_top_same_scores.extend(bot_top_same_scores)
                full_top_top_rev_scores.extend(top_top_rev_scores)
                full_bot_top_rev_scores.extend(bot_top_rev_scores)
        top_mean = np.mean([y/x for x,y in zip(full_top_top_same_scores, full_top_top_rev_scores)])
        bot_mean = np.mean([y/x for x,y in zip(full_bot_top_same_scores, full_bot_top_rev_scores)])
        print(f"top mean: {top_mean}")
        print(f"bot mean: {bot_mean}")


run_numerical_analysis()
