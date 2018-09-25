  # show all flags and its default values
  for k,v in tf.flags.FLAGS.__flags.items():
  	  print(k,v.__dict__['default'])
