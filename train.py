# -*- coding: utf-8 -*-
"""
Created on Thu May 30 21:42:07 2019

@author: cm
"""

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import sys
root_path = os.path.abspath(os.path.dirname(__file__)).split('classifier_multi_label')[0]
#root_path = os.path.abspath(__file__).split('classifier_multi_label')[0]
print(root_path)
sys.path.append(root_path)

import numpy as np
import pandas as pd
# 显示所有行、列
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import tensorflow as tf
from classifier_multi_label.networks import NetworkAlbert
from classifier_multi_label.utils.classifier_utils import get_features,get_features_val,get_features_test
from classifier_multi_label.hyperparameters import Hyperparamters as hp
from classifier_multi_label.utils.utils import select,shuffle_one,time_now_string
from classifier_multi_label.utils.evaluate_utils import accuracy,presision,recall,measure_auc
class_=hp.label_vocabulary
num_classes=len(class_)

pwd = os.path.dirname(os.path.abspath(__file__))
MODEL = NetworkAlbert(is_training=True)


# Get data features
input_ids,input_masks,segment_ids,label_ids = get_features()
num_train_samples = len(input_ids)
indexs = np.arange(num_train_samples)               
num_batchs = int((num_train_samples - 1) /hp.batch_size) + 1
print('Number of batch:',num_batchs)

# Set up the graph 
saver = tf.train.Saver(max_to_keep=hp.max_to_keep)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Load model saved before
MODEL_SAVE_PATH = os.path.join(pwd, hp.file_save_model)
ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
if ckpt and ckpt.model_checkpoint_path:
     saver.restore(sess, ckpt.model_checkpoint_path)
     print('Restored model!')


with sess.as_default():
    # Tensorboard writer
    writer = tf.summary.FileWriter(hp.logdir, sess.graph)
    for i in range(hp.num_train_epochs):
        np.random.shuffle(indexs)
        train_lab = []
        train_prb = []
        for j in range(num_batchs-1):
            # Get ids selected
            i1 = indexs[j * hp.batch_size:min((j + 1) * hp.batch_size, num_train_samples)]
            
            # Get features
            input_id_ = select(input_ids,i1)
            input_mask_ = select(input_masks,i1)
            segment_id_ = select(segment_ids,i1)
            label_id_ = select(label_ids,i1)
            
            # Feed dict
            fd = {MODEL.input_ids: input_id_,
                  MODEL.input_masks: input_mask_,
                  MODEL.segment_ids:segment_id_,
                  MODEL.label_ids:label_id_}
            
            # Optimizer
            sess.run(MODEL.optimizer, feed_dict = fd)   
            
            # Tensorboard
            if j%hp.summary_step==0:
                summary,glolal_step = sess.run([MODEL.merged,MODEL.global_step], feed_dict = fd)
                writer.add_summary(summary, glolal_step) 
                
            # Save Model
            if j%(num_batchs//hp.num_saved_per_epoch)==0:
                if not os.path.exists(os.path.join(pwd, hp.file_save_model)):
                    os.makedirs(os.path.join(pwd, hp.file_save_model))                 
                saver.save(sess, os.path.join(pwd, hp.file_save_model, 'model'+'_%s_%s.ckpt'%(str(i),str(j))))

            # Log
            if j % hp.print_step == 0:
                fd = {MODEL.input_ids: input_id_,
                      MODEL.input_masks: input_mask_,
                      MODEL.segment_ids:segment_id_,
                      MODEL.label_ids:label_id_}
                loss,input_label,prb = sess.run([MODEL.loss,MODEL.label_ids,MODEL.probabilities], feed_dict = fd)
                train_lab.extend(input_label.tolist())
                train_prb.extend(prb.tolist())
                acc_class = np.mean(accuracy(input_label, prb, num_classes,hp.threshold))
                presision_class = np.mean(presision(input_label, prb, num_classes, hp.threshold))
                recall_class = np.mean(recall(input_label, prb, num_classes, hp.threshold))
                auc_class = measure_auc(input_label, prb, num_classes)
                auc_class = np.mean([i for i in auc_class if i != '--'])

                print('Time:%s, Epoch:%s, Batch number:%s/%s, Loss:%.5f,准确率:%.5f,精确率:%.5f,召回率:%.5f,auc:%s'%(time_now_string(),str(i),str(j),str(num_batchs),loss, \
                                                                                                 acc_class,presision_class,recall_class,auc_class))

        acc = accuracy(np.array(train_lab), np.array(train_prb), num_classes, hp.threshold)
        pres = presision(np.array(train_lab), np.array(train_prb), num_classes, hp.threshold)
        rec = recall(np.array(train_lab), np.array(train_prb), num_classes, hp.threshold)
        auc = measure_auc(np.array(train_lab), np.array(train_prb), num_classes)
        dics = {'类别': class_, '正确率_train': acc, '精确率_train': pres, '召回率_train': rec, 'auc_train': auc}
        df_auc_train = pd.DataFrame(dics)

        print("train set mean column auc:\n",
              df_auc_train[['类别', '正确率_train', '精确率_train', '召回率_train', 'auc_train']])

        input_ids_val, input_masks_val, segment_ids_val, label_ids_val = get_features_val()
        fd = {MODEL.input_ids: input_ids_val,
              MODEL.input_masks: input_masks_val,
              MODEL.segment_ids: segment_ids_val,
              MODEL.label_ids: label_ids_val}
        val_loss, val_label, val_prb = sess.run([MODEL.loss, MODEL.label_ids, MODEL.probabilities], feed_dict=fd)
        acc_val = accuracy(np.array(val_label), np.array(val_prb), num_classes, hp.threshold)
        pres_val = presision(np.array(val_label), np.array(val_prb), num_classes, hp.threshold)
        rec_val = recall(np.array(val_label), np.array(val_prb), num_classes, hp.threshold)
        auc_val = measure_auc(np.array(val_label), np.array(val_prb), num_classes)
        print('Loss_val:%.5f'%(val_loss))
        dics = {'类别': class_, '正确率_val': acc_val, '精确率_val': pres_val, '召回率_val': rec_val, 'auc_val': auc_val}
        df_auc_train = pd.DataFrame(dics)

        print("val set mean column auc:\n",
              df_auc_train[['类别', '正确率_val', '精确率_val', '召回率_val', 'auc_val']])

        input_ids_test, input_masks_test, segment_ids_test, label_ids_test= get_features_test()
        fd = {MODEL.input_ids: input_ids_test,
              MODEL.input_masks: input_masks_test,
              MODEL.segment_ids: segment_ids_test,
              MODEL.label_ids: label_ids_test}
        test_loss, test_label, test_prb = sess.run([MODEL.loss, MODEL.label_ids, MODEL.probabilities], feed_dict=fd)
        acc_test = accuracy(np.array(test_label), np.array(test_prb), num_classes, hp.threshold)
        pres_test = presision(np.array(test_label), np.array(test_prb), num_classes, hp.threshold)
        rec_test = recall(np.array(test_label), np.array(test_prb), num_classes, hp.threshold)
        auc_test = measure_auc(np.array(test_label), np.array(test_prb), num_classes)
        print('Loss_val:%.5f' % (test_loss))
        dics = {'类别': class_, '正确率_test': acc_test, '精确率_test': pres_test, '召回率_test': rec_test, 'auc_test': auc_test}
        df_auc_train = pd.DataFrame(dics)

        print("test set mean column auc:\n",
              df_auc_train[['类别', '正确率_test', '精确率_test', '召回率_test', 'auc_test']])

    print('Train finished')
    
    
    
    




