import tensorflow as tf
import smodel_lib_v2 as model_lib_v2
#from object_detection import model_lib_v2
# model_lib_v2_path: C:\Users\peiwe\anaconda3\Lib\site-packages\object_detection
r"""
'pipeline_config_path' --> user_input, Path to pipeline config 
'num_train_steps' --> None, Number of train steps.
'eval_on_train_data' --> False, Enable evaluating on train, data (only supported in distributed training)
'sample_1_of_n_eval_examples' --> None, 'Will sample one of every n eval input examples, where n is provided.
'sample_1_of_n_eval_on_train_examples' --> 5, 'Will sample one of every n train input examples for evaluation, where n is provided. 
                                           This is only used if `eval_training_data` is True.
'model_dir' --> None, 'Path to output model directory where event and checkpoint files will be written.
'checkpoint_dir' --> None, Path to directory holding a checkpoint.  If `checkpoint_dir` is provided, this binary operates in eval-only mode
                     and writing resulting metrics to `model_dir`.
'eval_timeout' --> 3600, Number of seconds to wait for an evaluation checkpoint before exiting.
'use_tpu' --> False, Whether the job is executing on a TPU.
'tpu_name' --> default=None, Name of the Cloud TPU for Cluster Resolvers.
'num_workers' --> 1, When num_workers > 1, training uses 'MultiWorkerMirroredStrategy'. When num_workers = 1 it uses 'MirroredStrategy'
'checkpoint_every_n' --> 1000, Integer defining how often we checkpoint.
'record_summaries' --> True, Whether or not to record summaries defined by the model or the training pipeline. 
                       This does not impact the summaries of the loss values which are always recorded.
"""


class train_or_evaluate_model():
    def __init__(self, model_dir, pipeline_config_path):
        self.model_dir = model_dir
        self.pipeline_config_path = pipeline_config_path
        self.num_train_steps = 5000
        self.eval_on_train_data = False
        self.sample_1_of_n_eval_examples = None
        self.sample_1_of_n_eval_on_train_examples = 5
        self.checkpoint_dir = None
        self.eval_timeout = 3600
        self.use_tpu = False
        self.tpu_name = None
        self.num_workers = 1
        self.checkpoint_every_n = 1000
        self.record_summaries = True
        
    def start(self):
      tf.config.set_soft_device_placement(True)

      if self.checkpoint_dir:
        model_lib_v2.eval_continuously(
            pipeline_config_path=self.pipeline_config_path,
            model_dir=self.model_dir,
            train_steps=self.num_train_steps,
            sample_1_of_n_eval_examples=self.sample_1_of_n_eval_examples,
            sample_1_of_n_eval_on_train_examples=(
                self.sample_1_of_n_eval_on_train_examples),
            checkpoint_dir=self.checkpoint_dir,
            wait_interval=300, timeout=self.eval_timeout)
      else:
        if self.use_tpu:
          # TPU is automatically inferred if tpu_name is None and
          # we are running under cloud ai-platform.
          resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
              self.tpu_name)
          tf.config.experimental_connect_to_cluster(resolver)
          tf.tpu.experimental.initialize_tpu_system(resolver)
          strategy = tf.distribute.experimental.TPUStrategy(resolver)
        elif self.num_workers > 1:
          strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        else:
          strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
          model_lib_v2.train_loop(
              pipeline_config_path=self.pipeline_config_path,
              model_dir=self.model_dir,
              train_steps=self.num_train_steps,
              use_tpu=self.use_tpu,
              checkpoint_every_n=self.checkpoint_every_n,
              record_summaries=self.record_summaries)




