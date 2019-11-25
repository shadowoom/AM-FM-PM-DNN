import tensorflow_hub as tfhub
import tensorflow as tf
import tensorflow_text  # required for tokenization ops
import codecs
import numpy as np
from tqdm import tqdm

NUMBER_OF_SUBMISSIONS = 21

def encode_context(dialogue_history):
    """Encode the dialogue context to the response ranking vector space.

    Args:
        dialogue_history: a list of strings, the dialogue history, in
            chronological order.
    """

    # The context is the most recent message in the history.
    context = dialogue_history[-1]

    extra_context = list(dialogue_history[:-1])
    extra_context.reverse()
    extra_context_feature = " ".join(extra_context)

    return sess.run(
        context_encoding_tensor,
        feed_dict={
            text_placeholder: [context],
            extra_text_placeholder: [extra_context_feature],
        }
    )[0]

def encode_responses(texts):
    return sess.run(response_encoding_tensor, feed_dict={text_placeholder: texts})


def rank_per_conversation(context, candidates):
    context_encoding = encode_context(list(context))
    response_encodings = encode_responses(candidates)
    scores = context_encoding.dot(response_encodings.T)
    return scores
# context_encoding = encode_context(
#
#    [
#     "I'm looking for good courses to take.",
#     "Are you looking for courses in a specific area?",
#     "Not in particular.",
#     ]
# )

#candidate_responses = [
#    "Me neither.",
#    # We hope it selects the following sentence:
#    "Are you looking to take a very difficult class?",
#    "Why not?",
#    "Please finish the exercise first.",
#    "Nothing in particular?",
#    "Would you like fries with that?",
#    "School's for fools, look at me!",
#    "Our higher education system is one of the things that makes America exceptional.",
#]
#response_encodings = encode_responses(candidate_responses)
#
#print(
#    f"Best response:\n\t'{candidate_responses[top_idx]}'\n"
#    f"\t\tscore: {scores[top_idx]:.3f}"
#)
#
#indices = np.argsort(scores)[::-1][1:]
#print("Other responses:")
#for index in indices:
#    print(
#        f"\t'{candidate_responses[index]}'\n"
#        f"\t\tscore: {scores[index]:.3f}"
#    )
#

if __name__=='__main__':

    sess = tf.InteractiveSession(graph=tf.Graph())
    module = tfhub.Module("http://models.poly-ai.com/multi_context_convert/v1/model.tar.gz")
    text_placeholder = tf.placeholder(dtype=tf.string, shape=[None])
    extra_text_placeholder = tf.placeholder(dtype=tf.string, shape=[None])
    # The encode_context signature now also takes the extra context.
    context_encoding_tensor = module(
        {
            'context': text_placeholder,
            'extra_context': extra_text_placeholder,
        },
        signature="encode_context"
    )
    response_encoding_tensor = module(text_placeholder, signature="encode_response")
    encoding_dim = int(context_encoding_tensor.shape[1])
    print(f"ConveRT encodes contexts & responses to {encoding_dim}-dimensional vectors")
    sess.run(tf.tables_initializer())
    sess.run(tf.global_variables_initializer())
    # read test file
    test_file_path = '../../../data/twitter_data/final_test.txt'
    with codecs.open(test_file_path, mode='r', encoding='utf-8') as rf:
        test_lines = rf.readlines()
    test_lines = [line.strip().split('\t')[1:] for line in test_lines]
    conversation = []
    for i, line in enumerate(test_lines):
        if i == 0:
            context = line[:-1]
            candidates = [line[-1]]
            continue
        if i % NUMBER_OF_SUBMISSIONS == 0 and i != 0:
            conversation.append((context, candidates))
            context = line[:-1]
            candidates = [line[-1]]
            continue
        candidates.append(line[-1])
    conversation.append((context, candidates))
    test_scores = []
    for k, v in tqdm(conversation):
        score = rank_per_conversation(k, v).tolist()
        test_scores.extend(score)

    print("---------------------writing utterance level confidence score---------------------------")
    system_level_scores = {}
    for i in range(NUMBER_OF_SUBMISSIONS):
        system_level_scores[i] = []
    with codecs.open("context_am_ranking_score.txt", mode='w', encoding='utf-8') as wf:                                                                                                             
        wf.truncate()
    for i, score in enumerate(test_scores):
        system_level_scores[i % NUMBER_OF_SUBMISSIONS].append(score)
        with codecs.open("context_am_ranking_score.txt", mode='a', encoding='utf-8') as wf:
            wf.write(str(score) + '\n')
        if i % NUMBER_OF_SUBMISSIONS == NUMBER_OF_SUBMISSIONS - 1:
            with codecs.open("context_am_ranking_score.txt", mode='a', encoding='utf-8') as wf:
                wf.write('\n')
    print("---------------------Done writing utterance level confidence score ---------------------")


    print("---------------------Writing system level confidence score -----------------------------")
    with codecs.open("context_am_ranking_score_system_level.txt", mode='w', encoding='utf-8') as wf:
        wf.truncate()
    for k, v in system_level_scores.items():
        avg_score = sum(v) / len(v)
        with codecs.open("context_am_ranking_score_system_level.txt", mode='a', encoding='utf-8') as wf:
            wf.write(str(avg_score) + '\n')
    print("---------------------Done writing system level confidence score ------------------------")
    
    sess.close()
