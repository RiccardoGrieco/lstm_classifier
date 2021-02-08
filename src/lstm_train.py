#!/usr/bin/env python

import rospy
import numpy as np

import tensorflow
from tensorflow.keras.models import Sequential #load_model, model_from_json
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import Callback
from dataset_util import *

if tensorflow.__version__ >= "2":
    from tensorflow.compat.v1 import Graph, Session
else:
    from tensorflow import Graph, Session

import sys, os

# used to check if an error
# occurs in the reading of parameters
error_occurred = 0
def eprint(err):
    global error_occurred
    sys.stderr.write("ERROR: %s\n" % err)
    error_occurred = error_occurred + 1

MISSING_PARAMETER_STR = "Missing parameter: "

# trainer class
class lstm_train:
    ADMISSIBLE_ACTIVATION_FUNCTIONS_LIST = ['relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu', 'elu', 'exponential']

    SAVE_PREDICTIONS_NONE = 0
    SAVE_PREDICTIONS_BEST = 1
    SAVE_PREDICTIONS_ALL  = 2

    def __init__(self):
        rospy.init_node("lstm_train", anonymous=False)
        self.node_name = rospy.get_name()

        # dataset parameters
        self.dataset_train_file        = rospy.get_param("%s/dataset_train_file" % self.node_name, None)
        self.dataset_test_file         = rospy.get_param("%s/dataset_test_file" % self.node_name, None)
        self.num_input                 = rospy.get_param("%s/num_input" % self.node_name, None)
        self.num_classes               = rospy.get_param("%s/num_classes" % self.node_name, None)
        self.timesteps                 = rospy.get_param("%s/timesteps" % self.node_name, None)
        # model parameters       
        self.hid_layer_dim_list        = rospy.get_param("%s/hid_layer_dim_list" % self.node_name, None)
        self.hid_layer_activation_list = rospy.get_param("%s/hid_layer_activation_list" % self.node_name, None) 
        # train parameters       
        self.train_windows             = rospy.get_param("%s/train_windows" % self.node_name, [self.timesteps])
        self.max_epochs                = rospy.get_param("%s/max_epochs" % self.node_name, None)
        self.start_epochs              = rospy.get_param("%s/start_epochs" % self.node_name, self.max_epochs)
        self.epochs_steps              = rospy.get_param("%s/epochs_steps" % self.node_name, self.max_epochs)
        self.validation_fraction       = rospy.get_param("%s/validation_fraction" % self.node_name, 0.)
        self.keras_verbosity           = rospy.get_param("%s/keras_verbosity" % self.node_name, 0)
        self.keras_optimizer           = rospy.get_param("%s/keras_optimizer" % self.node_name, "adam")
        # evaluation parameters
        self.evaluation_start          = rospy.get_param("%s/evaluation_start" % self.node_name, 0)
        self.score_threshold           = rospy.get_param("%s/score_threshold" % self.node_name, 0)
        self.score_sequence_threshold  = rospy.get_param("%s/score_sequence_threshold" % self.node_name, 1)
        # directories
        self.models_directory           = rospy.get_param("%s/models_directory" % self.node_name, './models')
        self.model_name                = rospy.get_param("%s/model_name" % self.node_name, 'model')
        self.save_all_models           = rospy.get_param("%s/save_all_models" % self.node_name, False)
        # save classifications results
        self.save_test_classifications = rospy.get_param("%s/save_test_classifications" % self.node_name, 0)
        self.input_names               = rospy.get_param("%s/input_names" % self.node_name, None)
        self.classification_labels     = rospy.get_param("%s/classification_labels" % self.node_name, None)

        if self.check_parameters():
            print("Parameters have been correctly loaded\n")
        else:
            exit(0)

    # checks the consistency of the parameters
    def check_parameters(self):
        if self.dataset_train_file is None:
            eprint(MISSING_PARAMETER_STR + "dataset_train_file")
        elif not isinstance(self.dataset_train_file, str):
            eprint('"dataset_train_file must" be a string') 
        elif not os.path.isfile(self.dataset_train_file):
            eprint('The provided parameter "dataset_train_file" is not a valid file path')

        if self.dataset_test_file is None:
            eprint(MISSING_PARAMETER_STR + "dataset_test_file")
        elif not isinstance(self.dataset_test_file, str):
            eprint('"dataset_test_file must" be a string') 
        elif not os.path.isfile(self.dataset_test_file):
            eprint('The provided parameter "dataset_test_file" is not a valid file path')

        if self.num_input is None:
            eprint(MISSING_PARAMETER_STR + "num_input")
        elif not isinstance(self.num_input, int):
            eprint('"num_input" must be an integer value')
        elif self.num_input <= 0:
            eprint('"num_input" must be a positive integer')
        
        if self.num_classes is None:
            eprint(MISSING_PARAMETER_STR + "num_classes")
        elif not isinstance(self.num_classes, int):
            eprint('"num_classes" must be an integer value')
        elif self.num_classes <= 1:
            eprint('"num_classes" must be > 1')
    
        if self.timesteps is None:
            eprint(MISSING_PARAMETER_STR+"timesteps")
        elif not isinstance(self.timesteps,int) or self.timesteps<=0:
            eprint('"timesteps" must be a positive integer')

        if self.hid_layer_dim_list is None:
            eprint(MISSING_PARAMETER_STR + "hid_layer_dim_list")
        elif not isinstance(self.hid_layer_dim_list, list):
            eprint('"hid_layer_dim_list" must be a list')
        elif len(self.hid_layer_dim_list)==0:
            eprint('"hid_layer_dim_list" must be non-empty')
        elif not all([(str(x).isdigit() and int(str(x))>0) for x in self.hid_layer_dim_list]):
            eprint('All values in "hid_layer_dim_list" must be positive integers')
        else:
            self.hid_layer_dim_list = [int(x) for x in self.hid_layer_dim_list]

        if self.hid_layer_activation_list is None:
            eprint(MISSING_PARAMETER_STR + "hid_layer_activation_list")
        elif not isinstance(self.hid_layer_activation_list, list):
            eprint('"hid_layer_activation_list" must be a list')
        elif len(self.hid_layer_activation_list)==0:
            eprint('"hid_layer_activation_list" must be non-empty')
        elif len(self.hid_layer_dim_list)!=len(self.hid_layer_activation_list):
            eprint('"hid_layer_activation" must have the same number of elements as "hid_layer_dim_list"')
        else:
            self.hid_layer_activation_list = [str(x) for x in self.hid_layer_activation_list]
            if not all([x in lstm_train.ADMISSIBLE_ACTIVATION_FUNCTIONS_LIST for x in self.hid_layer_activation_list]):
                eprint('All functions in "hid_layer_activation_list" must be admissible for keras')

        if not isinstance(self.train_windows, list) or len(self.train_windows)==0:
            eprint('"train_windows" must a non-empty list')
        elif not all([(isinstance(x, int) and x>0 and self.timesteps%x==0) for x in self.train_windows]):
            eprint('"train_windows" values must divisors for "timesteps"')

        if self.max_epochs is None:
            eprint(MISSING_PARAMETER_STR + "max_epochs")
        elif not isinstance(self.max_epochs, int) or self.max_epochs<1:
            eprint('"max_epochs" must be a positive integer')

        if not isinstance(self.start_epochs, int) or self.start_epochs<0:
            eprint('"start_epochs" must be a non-negative integer')

        if not isinstance(self.epochs_steps, int) or self.epochs_steps<=0:
            eprint('"epochs_steps" must be a positive integer')

        if not isinstance(self.validation_fraction, float) or not 0.0<=self.validation_fraction<1:
            eprint('"validation_fraction" must be a floating point number in the interval [0,1[')

        if not isinstance(self.keras_verbosity, int) or not 0<=self.keras_verbosity<=2:
            eprint('"keras_verbosity" must be an integer in the interval 0..2')

        if not isinstance(self.evaluation_start, int) or self.evaluation_start<0 or self.evaluation_start>=self.timesteps:
            eprint('"evaluation_start" must be a non-negative integer not greater than "timesteps"')

        if not isinstance(self.score_threshold, float) or not(0<=self.score_threshold<1):
            eprint('"score_threshold" must be a floating point number in the interval [0,1[')

        if not isinstance(self.score_sequence_threshold, int) or not(0<self.score_sequence_threshold<self.timesteps):
            eprint('"score_sequence_threshold" must be an integer in the interval [1, "timesteps"[')
        
        if self.models_directory is None:
            eprint(MISSING_PARAMETER_STR + "models_directory")
        elif not isinstance(self.models_directory, str) or (os.path.exists(self.models_directory) and not os.path.isdir(self.models_directory)):
            eprint('"models_directory" must be an existing directory or an unused path')
        
        if self.model_name is None:
            eprint(MISSING_PARAMETER_STR + "model_name")
        elif not isinstance(self.model_name, str):
            eprint('"model_name" must be a string')

        if not isinstance(self.save_all_models, bool):
            eprint('"save_all_models" must be boolean')

        if not isinstance(self.save_test_classifications, int) or not 0<=self.save_test_classifications<=2:
           eprint('"save_tests_classifications" must be an integer value between 0 and 2')
        
        if self.input_names is None:
            self.input_names = ["C%d" % (i+1) for i in range(self.num_input)]
        elif not isinstance(self.input_names, list) or len(self.input_names)!=self.num_input:
            eprint('"input_names" must be a list of the size "num_input"')
        else:
            self.input_names = [str(x) for x in self.input_names] #type forcing

        if self.classification_labels is None:
            self.classification_labels = ["C%d" % (i+1) for i in range(self.num_classes)]
        elif not isinstance(self.classification_labels, list) or len(self.classification_labels)!=self.num_classes:
            eprint('"classification_labels" must be a list of the size "num_classes"')
        else:
            self.classification_labels = [str(x) for x in self.classification_labels] #type forcing

        return True

    class ResetOnBatchCallback(Callback):
        def on_batch_begin(self, batch, logs={}):
            if rospy.is_shutdown():
               exit(0)
            self.model.reset_states()

    class CheckROSCallback(Callback):
        def on_batch_begin(self, batch, logs=None):
            if rospy.is_shutdown():
                exit(0)

    class ResetOnEpochCallback(Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.model.reset_states()

    # train the networks
    def train_lstm(self):
        print("Loading dataset...")
        dataset_train_x, dataset_train_y = load_dataset(self.dataset_train_file, self.num_input, self.num_classes)
        dataset_test_x, dataset_test_y   = load_dataset(self.dataset_test_file, self.num_input, self.num_classes)
        train_dim = len(dataset_train_x)
        print("Dataset loaded")

        best_accuracy = -1.
        best_model = None
        best_classification_flows = None

        # training callbacks
        callbacks = [lstm_train.ResetOnEpochCallback(), lstm_train.CheckROSCallback()]
        shuffle = False
        
        # initialize paths
        if not os.path.exists(self.models_directory):
            os.makedirs(self.models_directory)
        result_file = open(os.path.join(self.models_directory, "train_results"), 'w')
        result_file.write("learning_timesteps epochs accuracy\n")
        saved_models_root = os.path.join(self.models_directory, "trained")
        if not os.path.exists(saved_models_root):
            os.makedirs(saved_models_root)

        for ts in np.array(self.train_windows): 
            train_x, train_y, test_x, test_y = divide_data(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y, 
                                                                self.num_input, self.num_classes, ts)
            
            model, graph, session = self.build_keras_model_training(dataset_train_x)
            best_in_current_session = False
            
            epochs = 0
            loop_epochs = self.start_epochs #first cycle epochs

            while epochs<self.max_epochs:
                #train the network 
                print("-------  Training with "+str(ts) +" as timesteps and "+str(epochs+loop_epochs) + " epochs----------------")
                with graph.as_default(), session.as_default():
                    model.fit(train_x, train_y, epochs = loop_epochs, verbose=self.keras_verbosity, callbacks = callbacks,
                                            batch_size=train_dim, validation_split=self.validation_fraction, shuffle=shuffle)
                    #rebuild model for evaluation 
                    test_model = self.build_keras_model_online(model, graph, session)
                
                epochs = epochs + loop_epochs
                loop_epochs = self.epochs_steps

                classification_flows, classification_results, accuracy, cf, = self.evaluate_network(test_model, test_x, test_y, graph, session)
                print("Trained with accuracy = " + str(accuracy))
                result_file.write("%d %d %f\n" % (ts, epochs, accuracy))
                
                # save the trained model
                if self.save_all_models:
                    saved_model_directory = os.path.join(saved_models_root, "%d_%d" % (ts, epochs))
                    if not os.path.isdir(saved_model_directory):
                        os.makedirs(saved_model_directory)
                    if self.save_test_classifications == lstm_train.SAVE_PREDICTIONS_ALL:
                        self.save_classifications(classification_flows, classification_results, saved_model_directory, test_x, test_y)
                    self.save_model(test_model, graph, session, os.path.join(saved_model_directory, self.model_name))

                # update best network
                if accuracy>best_accuracy:
                    if not(best_model is None):
                        del best_model
                        if not(best_in_current_session):
                            best_session.close()
                            del best_session, best_graph
                    best_model = test_model
                    best_accuracy = accuracy
                    best_graph = graph
                    best_session = session
                    best_in_current_session = True

                    if not(best_classification_flows is None):
                        del best_classification_flows, best_classification_results

                    best_classification_flows = classification_flows
                    best_classification_results = classification_results

                else:
                    del test_model
                    del classification_flows, classification_results

                result_file.flush()

            self.save_model(best_model, best_graph, best_session, os.path.join(self.models_directory, self.model_name))
            if self.save_test_classifications >= lstm_train.SAVE_PREDICTIONS_BEST:
                self.save_classifications(best_classification_flows, best_classification_results, self.models_directory, test_x, test_y)
            
            del model, graph, session #delete training model
            del train_x, train_y, test_x, test_y

        result_file.close()
       
    # build the model for training
    def build_keras_model_training(self, train_x):
        #model building
        graph = Graph()
        with graph.as_default():
            session = Session()
            with session.as_default():
                model = Sequential()
                
                #first layer
                model.add(LSTM(self.hid_layer_dim_list[0], batch_input_shape=(len(train_x), None, self.num_input), stateful=True, recurrent_dropout=0.1, activation=self.hid_layer_activation_list[0]))

                #hidden layers
                for i in range(1, len(self.hid_layer_dim_list)):
                    model.add(LSTM(self.hid_layer_dim_list[i], recurrent_dropout=0.1, stateful=True, activation=self.hid_layer_activation_list[i]))

                #output layer                
                model.add(Dense(self.num_classes, activation='softmax'))

                #compile
                model.compile(loss='categorical_crossentropy', optimizer=self.keras_optimizer, metrics =['categorical_accuracy'])

        return model, graph, session
    
    #build keras model for online classification
    def build_keras_model_online(self, model, graph, session):
        with graph.as_default(), session.as_default():
            new_model = Sequential()

            #first layer
            new_model.add(LSTM(self.hid_layer_dim_list[0], batch_input_shape=(1, None, self.num_input), stateful=True, recurrent_dropout=0.1, activation=self.hid_layer_activation_list[0]))

            #hidden layers
            for i in range(1, len(self.hid_layer_dim_list)):
                new_model.add(LSTM(self.hid_layer_dim_list[i], recurrent_dropout=0.1, stateful=True, activation=self.hid_layer_activation_list[i]))
            
            #output layer                
            new_model.add(Dense(self.num_classes, activation='softmax'))

            for layer_prev, layer_new in zip(model.layers, new_model.layers):
                layer_weights = layer_prev.get_weights()
                layer_new.set_weights(layer_weights)

            new_model.compile(loss='categorical_crossentropy', optimizer=self.keras_optimizer, metrics =['categorical_accuracy'])
        return new_model

    #save a model
    def save_model(self, model, graph, session, model_path):
        try:
            with graph.as_default(), session.as_default():
                model.save(filepath=model_path, save_format='h5')
            return True  # Save successful
        except Exception as e:
            print("---------SAVE ERROR--------")
            print(str(e))
            return False

    def save_classifications(self, classifications, results, directory, test_x, test_y):
        if not(os.path.exists(directory)):
            os.makedirs(directory)
        
        summary_file = open(os.path.join(directory,'summary'), 'w')
        scores_files = open(os.path.join(directory,'scores'), 'w')
        
        #write headers
        summary_file.write('sequence expected result\n')
        classes_header = "".join([" "+class_name for class_name in self.classification_labels])
        scores_files.write('sequence'+classes_header+'\n')

        i = 0
        for classification_flow in classifications:
            #write scores
            for classification in classification_flow:
                scores_files.write(str(i).join([" "+str(score) for score in classification])+'\n')

            #write classification result
            summary_file.write("%d %d %s\n" % (i, test_y[i], results[i]))

            i = i +1

        summary_file.close()
        scores_files.close()

    #evaluate the online classifier
    def evaluate_network(self, model, test_x, test_y, graph=None, session=None):        
        classification_results = np.array([-1]*len(test_x)).astype(int)
        classifications = []

        with graph.as_default(), session.as_default():
            for i in range(len(test_x)):
                sequence = test_x[i]
                classification_flow = []
                last_class = -1
                class_count = 0 
                found = False

                model.reset_states()
                for j in range(0,len(sequence),self.num_input):
                    x = sequence[j:j+self.num_input]
                    p = model.predict(np.reshape(x,(1, 1, self.num_input)), batch_size=1)[0]
                    classification_flow.append(p)
                    if j/self.num_input<self.evaluation_start: #ignore first data
                        continue
                    current_class = int(np.argmax(p))
                    if p[current_class]>=self.score_threshold: # score must have min value
                        if current_class!=last_class:
                            class_count = 0
                            last_class=current_class
                        class_count = class_count+1
                        if class_count==self.score_sequence_threshold and not(found): # same classification for awhile
                            found = True
                            classification_results[i] = current_class
                
                classifications.append(classification_flow)
            model.reset_states()

        # confusion matrix
        cf = np.zeros((self.num_classes, self.num_classes))
        classes_dim = np.zeros(self.num_classes).astype(int)
        for i in range(len(classification_results)):
            if classification_results[i]!=-1:
                cf[classification_results[i]][test_y[i]] = cf[classification_results[i]][test_y[i]]+1
            classes_dim[test_y[i]] = classes_dim[test_y[i]]+1

        # accuracy
        accuracy = np.sum([1 if c==y else 0 for c,y in zip(classification_results, test_y)])/float(len(test_y))

        return classifications, classification_results, accuracy, cf

    def run(self):
        self.train_lstm()
        
        
if __name__ == "__main__":
    classifier = lstm_train()
    classifier.run()