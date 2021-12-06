""" Runs experiments with CorefModel.

Try 'python run.py -h' for more details.
"""

import argparse
from contextlib import contextmanager
import datetime
import random
import sys
import time
from convert_to_heads_talisman import *
import numpy as np  # type: ignore
import torch        # type: ignore

from coref import CorefModel


@contextmanager
def output_running_time():
    """ Prints the time elapsed in the context """
    start = int(time.time())
    try:
        yield
    finally:
        end = int(time.time())
        delta = datetime.timedelta(seconds=end - start)
        print(f"Total running time: {delta}")


def seed(value: int) -> None:
    """ Seed random number generators to get reproducible results """
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)           # type: ignore
    torch.backends.cudnn.deterministic = True   # type: ignore
    torch.backends.cudnn.benchmark = False      # type: ignore


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("mode", choices=("train", "eval","predict"))
    argparser.add_argument("experiment")
    argparser.add_argument("--config-file", default="config.toml")
    argparser.add_argument("--data-split", choices=("train", "dev", "test"),
                           default="test",
                           help="Data split to be used for evaluation."
                                " Defaults to 'test'."
                                " Ignored in 'train' mode.")
    argparser.add_argument("--batch-size", type=int,
                           help="Adjust to override the config value if you're"
                                " experiencing out-of-memory issues")
    argparser.add_argument("--warm-start", action="store_true",
                           help="If set, the training will resume from the"
                                " last checkpoint saved if any. Ignored in"
                                " evaluation modes."
                                " Incompatible with '--weights'.")
    argparser.add_argument("--weights",
                           help="Path to file with weights to load."
                                " If not supplied, in 'eval' mode the latest"
                                " weights of the experiment will be loaded;"
                                " in 'train' mode no weights will be loaded.")
    argparser.add_argument("--word-level", action="store_true",
                           help="If set, output word-level conll-formatted"
                                " files in evaluation modes. Ignored in"
                                " 'train' mode.")
    args = argparser.parse_args()
    if args.warm_start and args.weights is not None:
        print("The following options are incompatible:"
              " '--warm_start' and '--weights'", file=sys.stderr)
        sys.exit(1)
    seed(2020)
    model = CorefModel(args.config_file, args.experiment)
    if args.batch_size:
        model.config.a_scoring_batch_size = args.batch_size

    if args.mode == "train":
        if args.weights is not None or args.warm_start:
            model.load_weights(path=args.weights, map_location="cpu",
                               noexception=args.warm_start)
        with output_running_time():
            model.train()
    if args.mode == "predict":
        model.load_weights(path=args.weights, map_location="cpu",
                           ignore={"bert_optimizer", "general_optimizer",
                                   "bert_scheduler", "general_scheduler"})

        text = ["We","ca","n't","say","how","those","who","will","be","voting","on","Samuel","Alito","'s","nomination","to","Supreme","Court","feel","about","him","/.","but","we","can","say","something","about","how","you","feel","tonight","/.","Here","are","the","results","of","the","CNN","USA","Today","Gallup","poll","released","just","minutes","ago","of","those","asked","about","the","president","'s","choice","Judge","Alito","to","succeed","Justice","Sandra","Day","O'Connor","/.","Seventeen","percent","said","he","'s","an","excellent","choice","/.","twenty","-","six","percent","called","him","a","good","choice","/.","twenty","-","two","percent","rated","him","only","fair","/.","seventeen","percent","thought","he","was","a","poor","choice","/.","Questions","about","the","facts","or","what","were","presented","as","facts","that","led","the","United","States","into","the","war","in","Iraq","spilled","into","open","warfare","today","on","the","Senate","floor","/.","Democrats","forced","the","Senate","into","a","rare","closed","door","session","/.","Republican","leader","Bill","Frist","said","the","Senate","was","hijacked","/.","CNN","'s","Ed","Henry","was","there","/.","A","Democratic","sneak","attack","that","sent","shock","waves","through","the","Senate","/.","Mr.","President","enough","time","has","gone","by","/.","I","demand","on","behalf","of","the","American","people","that","we","understand","why","these","investigations","are","n't","being","conducted","/.","Democratic","leader","Harry","Reed","accused","Republicans","of","failing","to","probe","allegations","the","White","House","manipulated","intelligence","to","justify","the","war","in","Iraq","/.","And","in","accordance","with","rule","twenty","-","one","I","now","move","that","Senate","go","into","closed","session","/.","President","I","second","the","motion","/.","An","easy","but","rare","maneuver","with","extraordinary","consequences","/.","The","Senate","chamber","was","locked","down",",","television","cameras","shut","off",",","so","law","makers","could","go","into","secret","session","to","debate","/.","Republican","leader","Bill","Frist","was","enraged","/.","Not","with","the","previous","Democratic","leader","or","the","current","Democratic","leader","have","ever","I","been","slapped","in","the","face","with","such","an","affront","to","the","leadership","of","this","grand","institution","/.","There","has","been","at","least","consideration","for","the","other","side","of","the","aisle","before","a","stunt","/.","and","this","is","a","pure","stunt","/.","Reed","refused","to","back","down","demanding","the","Republican","led","intelligence","committee","finish","a","long","awaited","report","on","whether","the","Bush","administration","twisted","intelligence","/.","This","investigation","has","been","stymied","stopped",",","obstructions","thrown","up","every","step","of","the","way","/.","That","'s","the","real","slap","in","the","face","/.","that","'s","the","slap","in","the","face","/.","And","today","the","American","people","are","going","to","see","a","little","bit","of","light","/.","What","'s","really","going","on","is","Democrats","feel","emboldened","by","the","indictment","of","Vice","President","Cheney","'s","former","chief","of","staff","believing","this","is","their","chance","to","issue","a","broader","indictment","of","the","Bush","administration","/.","We","have","lost","over","two","thousand","of","our","best","and","bravest","/.","over","fifteen","thousand","have","been","seriously","wounded","/.","We","are","spending","more","than","six","million","dollars","a","month","with","no","end","in","sight","/.","and","this","Republican","led","Senate","intelligence","committee","refuses","to","even","ask","the","hard","questions","about","the","misinformation","/-","Republicans","insist","they","'re","completing","the","investigation","/.","and","this","is","just","a","distraction","/.","This","is","purely","political","/.","This","is","settling","an","old","political","score","/.","Democrats","say","they","also","want","to","signal","they","'re","ready","to","stand","up","to","the","Republican","majority","and","may","even","filibuster","the","president","'s","latest","Supreme","Court","pick","Samuel","Alito","a","move","that","would","make","these","events","seem","like","the","opening","fireworks","in","a","much","nastier","battle","/.","Ed","Henry","CNN","Capitol","Hill","/.","So","do","you","think","we","just","saw","the","outlines","to","what","the","midterm","election","battlelines","might","look","like","/?","A","debate","that","'s","likely","to","rage","on","for","many","many","months","to","come","/."]
        sent_ids = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11,11,11,11,11,12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,16,16,16,16,16,16,17,17,17,17,17,17,17,17,17,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,19,19,19,19,19,19,19,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,22,22,22,22,22,22,22,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,25,25,25,25,25,25,25,25,25,26,26,26,26,26,26,26,26,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,29,29,29,29,29,29,29,29,29,29,29,29,30,30,30,30,30,30,30,30,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,33,33,33,33,33,33,33,33,34,34,34,34,34,34,34,35,35,35,35,35,36,36,36,36,36,36,36,36,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,38,38,38,38,38,38,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40]

        doc = doc_from_text(text, sent_ids)

        model.predict(doc)
    else:
        model.load_weights(path=args.weights, map_location="cpu",
                           ignore={"bert_optimizer", "general_optimizer",
                                   "bert_scheduler", "general_scheduler"})
        model.evaluate(data_split=args.data_split,
                       word_level_conll=args.word_level)
