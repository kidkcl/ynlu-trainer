import sys
import io
import time

from pprint import pprint

from ynlu import IntentClassifierClient

def call_client(cla_id, cla_token):
    client = IntentClassifierClient(
        token = cla_token
    )
    client.set_classifier(classifier_id = cla_id)
    return client

def add_utt_success(client, match, utterance):
    pairObj = []
    myDict = {}
    myDict['intent'] = str(match)
    myDict['utterance'] = str(utterance)
    pairObj.append(myDict)
    client.add_intent_utterance_pairs(pairObj)
    return True

def main():
    cla_id = ''
    cla_token = ''
    utterance = ''
    command = ''
    # input: your classifier id
    cla_id = input(">>> Your classifier id: ")
    # input: your token
    cla_token = input(">>> Your token: ")
    client = call_client(cla_id, cla_token)
    print(">>> Your classifier id is " + client.classifier_id)
    while True:
        # To stop training and predict, enter 'train_stop'
        utterance = input(">>> Test text: ")
        if utterance == 'train_stop':
            break
        result = client.predict(utterance)
        # pprint(result)
        rank_result = sorted(result, key=lambda x: x['score'])
        pprint(rank_result)
        case = input(">>> Is the result correct? Yes(1) No(2)")
        if int(case) == 2:
            match = input(">>> Enter the intent this utterance should match to: ")
            print("Adding utterance to that intent... \n")
            while not add_utt_success(client, match, utterance):
                print("Adding to " + match + " failed!! \n")
                match = input(">>> Enter the intent again: ")
            client.train()
            print("Training...")
            while True:
                if not client.classifier_is_traning():
                    break
                time.sleep(3)
        else:
            pass

if __name__ == '__main__':
    main()
