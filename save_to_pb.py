from absl import app, flags, logging
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

import effnetv2_model

FLAGS = flags.FLAGS

def define_flags():
  """Define all flags for binary run."""
  flags.DEFINE_string('mode', 'eval', 'Running mode.')
  flags.DEFINE_string('image_path', None, 'Location of test image.')
  flags.DEFINE_integer('image_size', None, 'Image size.')
  flags.DEFINE_string('model_dir', None, 'Location of the checkpoint to run.')
  flags.DEFINE_string('model_name', 'efficientnetv2-b0', 'Model name to use.')
  flags.DEFINE_string('dataset_cfg', 'Imagenet', 'dataset config name.')
  flags.DEFINE_string('hparam_str', '', 'k=v,x=y pairs or yaml file.')
  flags.DEFINE_bool('debug', False, 'If true, run in eager for debug.')
  flags.DEFINE_string('export_dir', None, 'Export or saved model directory')
  flags.DEFINE_string('trace_file', '/tmp/a.trace', 'If set, dump trace file.')
  flags.DEFINE_integer('batch_size', 16, 'Batch size.')
  flags.DEFINE_bool('mixed_precision', False, 'If True, use mixed precision.')

def build_tf2_model():
  """Build the tf2 model."""
  tf.config.run_functions_eagerly(FLAGS.debug)
  if FLAGS.mixed_precision:
    # Use 'mixed_float16' if running on GPUs.
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

  model = effnetv2_model.get_model(
      FLAGS.model_name,
      FLAGS.hparam_str,
      include_top=True,
      weights=FLAGS.model_dir or 'imagenet')
  model.summary()
  return model

def main(_) -> None:
    model = build_tf2_model() #build efficientnetv2 model 
    input = tf.keras.Input(shape=(224,224,3), batch_size=1) 

    keras_model = tf.keras.Model(inputs=[input], outputs=tf.nn.softmax(model.call(input, training=False))) #keras model
    keras_model.save('./efficientnetv2-b0_saved_model', save_format='tf') #save to tf saved model


if __name__ == '__main__':
  logging.set_verbosity(logging.ERROR)
  define_flags()
  app.run(main)