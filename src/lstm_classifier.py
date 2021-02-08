#!/usr/bin/env python

import threading
import rospy
import numpy as np
import os
import prova.srv
from prova.srv import classifyResponse

import tensorflow
import tensorflow.keras.models

if tensorflow.__version__ >= "2":
    from tensorflow.compat.v1 import Graph, Session
else:
    from tensorflow import Graph, Session

class NetworksEntry:
    def _init_(self, net=None, graph=None, session=None):
        self.net = net
        self.graph = graph
        self.session = session


# training on linked windows (no reset of states)
class LstmClassifier:

    result_error_messages = {1 : 'ID Already used',
                             2 : 'No model for provided path',
                             3 : 'Invalid model format for provided path',
                             4 : 'The provided Network ID is not allocated',
                             5 : 'The size of the provided input is not compatible with the network',
                             6 : 'Action not recognized'}

    def __init__(self):
        rospy.init_node("LstmClassifier", anonymous=False)
        print("name: "+rospy.get_name())
        self.node_name = rospy.get_name()
        
        self.service_name = rospy.get_param(self.node_name+"/service_name", "lstm_classifier/service")

        if not self.check_parameters():
            print("Cannot create the service")
            exit(1)

        # allocated networks
        self.networks = dict()
        self.networks_lock = threading.Lock()
        
        self.classificationService = rospy.Service(self.service_name, prova.srv.classify, self.classify_service_handler)
        print("Service ready")

    def check_parameters(self):
        parameters_ok = True

        if self.service_name is None:
            print("Missing parameter: service_name")
            parameters_ok = False
        elif not isinstance(self.service_name, str) or not self.service_name:
            print('Parameter "service_name" must be a non-empty string')
            parameters_ok = False

        return parameters_ok

    def print_classify_error(self, result):
        print(LstmClassifier.result_error_messages[result])

    # allocate a new network
    def allocate(self, request):
        response = classifyResponse()
        response.result = classifyResponse.OK
        print("Allocating network %s from path %s\n" % (request.net_id, request.model_path))

        # allocate network if ID is not occupied
        self.networks_lock.acquire()
        if request.net_id in self.networks:
            response.result = classifyResponse.ID_OCCUPIED
        else:
            entry = NetworksEntry()
            self.networks[request.net_id] = entry
        self.networks_lock.release()

        # if id is not occupied create the network
        if response.result == classifyResponse.OK:
            graph = Graph()
            with graph.as_default():
                session = Session(graph=graph)
                with session.as_default():
                    if os.path.exists(request.model_path):
                        try:
                            model = tensorflow.keras.models.load_model(filepath=request.model_path, compile=True)
                        except:
                            response.result = classifyResponse.MODEL_NOT_FOUND
                    else:
                        response.result = classifyResponse.INVALID_MODEL

        # if no error has occurred hook the created network to the dictionary
        if response.result == classifyResponse.OK:
            entry.graph = graph
            entry.session = session
            entry.net = model
            print("Successfully created network %s\n" % (request.net_id))
        else:
            self.networks_lock.acquire()
            del self.networks[request.net_id]
            self.networks_lock.acquire()
            del session, graph
            print("Error in creating network %s\n" % (request.net_id))

        return response

    # classify a requested input
    def classify(self, request):
        response = classifyResponse()
        response.result = classifyResponse.OK
        
        # get the network
        self.networks_lock.acquire()
        if request.net_id in self.networks:
            entry = self.networks[request.net_id]
        else:
            response.result = classifyResponse.NET_NOT_ALLOCATED
        self.networks_lock.release()

        # classify using the network if found
        if response.result == classifyResponse.OK:
            try:
                with entry.graph.as_default(), entry.session.as_default():               
                    output = entry.net.predict(np.reshape(np.array(request.inputs), (1,1,len(request.inputs))), batch_size=1)[0]
                
                i = np.argmax(output)
                response.c = i
                response.score = output[i]
            except ValueError:
                response.result = classifyResponse.INCORRECT_INPUT_SIZE
        
        return response 

    # resets the state of a network
    def reset(self, request):
        response = classifyResponse()
        response.result = classifyResponse.OK

        # look for the network
        self.networks_lock.acquire()
        if request.net_id in self.networks:
            entry = self.networks[request.net_id]
        else:
            response.result = classifyResponse.NET_NOT_ALLOCATED
        self.networks_lock.release()

        # if found, reset the internal state
        if response.result == classifyResponse.OK:
            with entry.graph.as_default(), entry.session.as_default():
                entry.net.reset_states()
            
        return response

    # delete the allocated LSTM
    def delete(self, request):
        response = classifyResponse()
        response.result = classifyResponse.OK

        # access network list
        self.networks_lock.acquire()
        if request.net_id in self.networks:
            entry = self.networks[request.net_id]
            del self.networks[request.net_id]  
        else:
            response.result = classifyResponse.NET_NOT_ALLOCATED
        self.networks_lock.release()

        # if found, delete the network
        if response.result == classifyResponse.OK:
            entry.session.close()
            del entry.session, entry.graph, entry.net

        return response

    # handler for the service
    def classify_service_handler(self, request):
        response = prova.srv.classifyResponse()
        action = int(request.action)

        # call the corresponding
        if action == request.ALLOCATE:
            response = self.allocate(request)
        elif action == request.CLASSIFY:
            response = self.classify(request)
        elif action == request.RESET:
            response = self.reset(request)
        elif action == request.DELETE:
            response = self.delete(request)
        else:
            response.result = response.ACTION_UNKNOWN

        if response.result != classifyResponse.OK:
            self.print_classify_error(response.result)

        return response

if __name__ == "__main__":
    classifier = LstmClassifier()
    rospy.spin()