import Resnet50_siamese_network as r50sn

network = r50sn.Resnet50_siamese_network(save_path="", input_shape=(224, 224, 3))
network.define_Model_architecture()
network.compile_the_model()
network.show_model_summary()