import random
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
import pickle
from collections import Counter

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def kaggle_classify():
    train_x, train_y = load_data()

    # Get the feature set 
    feature_set = get_feature_set()

    print 'Feature set: ', feature_set
    filter_train_x = extract_feature(feature_set, train_x)
    filter_train_y = train_y
    print 'Finished filtering...'


    n_estimators = 100
    learning_rate = .1
    max_depth = None
    random_state = 0

    # Run cross validation for our list of classifiers
    run_cross_validation(filter_train_x, filter_train_y)


    # List of classifiers to combine.
    # If you want to add a classifierl make sure to also add it to 
    # the run_cross_validation method
    print 'Processing first classifier'
    clf_1 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), n_estimators=400, learning_rate=.5, 
        random_state=0).fit(filter_train_x, filter_train_y)

    print 'Processing second classifier'
    # clf_2 = KNeighborsClassifier(n_neighbors=5).fit(filter_train_x, filter_train_y)
    clf_2 = RandomForestClassifier(n_estimators=50).fit(filter_train_x, filter_train_y)
    # clf_2 = ExtraTreesClassifier(n_estimators=100, max_depth=None, 
    #     random_state=0).fit(filter_train_x, filter_train_y)

    print 'Processing third classifier'
    clf_3 = GradientBoostingClassifier(n_estimators=400, learning_rate=1, 
        max_depth=1, random_state=0).fit(filter_train_x, filter_train_y)

    # ADD NEW CLASSIFIERS HERE




    # Combination of all those classifiers
    print 'Processing total classifier'
    # ADD NEW CLASSIFIERS HERE
    total_clf = VotingClassifier(estimators=[('clf_1', clf_1), ('clf_2', clf_2), ('clf_3', clf_3)], 
        voting='hard').fit(filter_train_x, filter_train_y)

    print 'Finished classfying...'


    # Total Training Error
    print 'Total Training accuracy: ', gen_clf_accuracy(clf_1, clf_2, clf_3, filter_train_x, filter_train_y)
    print 'Total Training accuracy: ', total_clf.score(filter_train_x, filter_train_y)


    # Processing and filtering test data
    test_x = process_test_data('test_2008.csv')
    filter_test_x = extract_feature(feature_set, test_x)
    print len(filter_test_x)

    # Predicting test data
    # pred = gen_mult_clf_prediction(clf_1, clf_2, clf_3, filter_test_x)
    pred = total_clf.predict(filter_test_x)
    print pred
    print 'Fraction of 1s: ', sum([1 if x == 1 else 0 for x in pred]) / float(len(pred))
    gen_file(pred)



def gen_clf_accuracy(clf_1, clf_2, clf_3, train_x, train_y):
    '''
    Returns accuracy using majority rule
    '''
    assert len(train_x) == len(train_y)

    # Total Training Error
    train_pred_1 = clf_1.predict(train_x)
    train_pred_2 = clf_2.predict(train_x)
    train_pred_3 = clf_3.predict(train_x)

    correct = 0
    majority_rule = 0
    for i, x in enumerate(train_y):
        c = Counter([train_pred_1[i], train_pred_2[i], train_pred_3[i]])
        val, count = c.most_common()[0]
        if count != 3:
            majority_rule += 1
        if val == x:
            correct += 1

    print 'Prediction difference: ', float(majority_rule) / len(train_y)
    return float(correct) / len(train_y)


# def gen_mult_clf_prediction(clf_1, clf_2, clf_3, train_x):
#     # Total Training Error
#     train_pred_1 = clf_1.predict(train_x)
#     train_pred_2 = clf_2.predict(train_x)
#     train_pred_3 = clf_3.predict(train_x)

#     pred = []
#     for i, x in enumerate(train_x):
#         c = Counter([train_pred_1[i], train_pred_2[i], train_pred_3[i]])
#         val = c.most_common()[0][0]
#         pred.append(val)
#     return np.asarray(pred)


def run_cross_validation(master_train_x, master_train_y):
    # Store training and validation error for each N
    kf = KFold(n_splits=6)
    total_train_err = 0
    total_valid_err = 0
    for train_index, test_index in kf.split(master_train_x):
        print 'Fold train index: ', train_index, len(train_index)
        print 'Fold test index: ', test_index, len(test_index)
        # Split validation data
        x_train, x_test = master_train_x[train_index], master_train_x[test_index]
        y_train, y_test = master_train_y[train_index], master_train_y[test_index]


        # List of classifiers
        # Should be exactly the same as the ones in kaggle_classify

        # Hot classifier that gives good results
        print 'Processing first classifier'
        clf_1 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), n_estimators=400, learning_rate=.5, 
            random_state=0).fit(x_train, y_train)

        print 'Processing second classifier'
        # clf_2 = KNeighborsClassifier(n_neighbors=3).fit(x_train, y_train)
        clf_2 = RandomForestClassifier(n_estimators=50).fit(x_train, y_train)
        # clf_2 = ExtraTreesClassifier(n_estimators=100, max_depth=None, 
        #     random_state=0).fit(x_train, y_train)


        print 'Processing third classifier'
        clf_3 = GradientBoostingClassifier(n_estimators=400, learning_rate=1, 
            max_depth=1, random_state=0).fit(x_train, y_train)


        # ADD NEW CLASSIFIERS HERE


        print 'Processing total classifier'
        # ADD NEW CLASSIFIERS HERE
        total_clf = VotingClassifier(estimators=[('clf_1', clf_1), ('clf_2', clf_2), ('clf_3', clf_3)], 
            voting='hard').fit(x_train, y_train)



        # Prints out the error rate for each classifier,
        # ADD NEW CLASSIFIERS HERE
        for index, clf in enumerate([clf_1, clf_2, clf_3]):
            print 'Output of classifier: ', index
            # clf_err_train = compute_error_clf(clf, x_train, y_train)
            clf_err_train = 1 - clf.score(x_train, y_train)
            print 'Training error per fold: ', clf_err_train

            # Add testing error
            # clf_err_test = compute_error_clf(clf, x_test, y_test)
            clf_err_test = 1 - clf.score(x_test, y_test)
            print 'Valid error per fold: ', clf_err_test
            


        # Computes the training and validation error of total classifier
        err_train = 1 - total_clf.score(x_train, y_train)
        total_train_err += err_train
        print 'Final Training error per fold: ', err_train

        # Add testing error
        err_test = 1 - total_clf.score(x_test, y_test)
        total_valid_err += err_test
        print 'Final Valid error per fold: ', err_test




    # Store averages of 5 folds
    avg_train_err = total_train_err / float(kf.get_n_splits(master_train_x))
    avg_valid_err = total_valid_err / float(kf.get_n_splits(master_train_x))
    print 'Average training error per point: ', avg_train_err
    print 'Average validation error per point: ', avg_valid_err




def load_data():
    # train_x, train_y = process_train_data('train_2008.csv')

    # # Dump training and testing onto a file
    # train_x_pickle = open('train_x_pickle', 'w')
    # pickle.dump(train_x, train_x_pickle)
    # train_x_pickle.close()
    # train_y_pickle = open('train_y_pickle', 'w')
    # pickle.dump(train_y, train_y_pickle)
    # train_y_pickle.close()

    # Load training and testing onto file
    train_x_pickle = open('train_x_pickle', 'r')
    train_x = pickle.load(train_x_pickle)
    train_x_pickle.close()
    train_y_pickle = open('train_y_pickle', 'r')
    train_y = pickle.load(train_y_pickle)
    train_y_pickle.close()
    print 'Finished loading data....'
    return train_x, train_y

def create_neural_network(X_train, y_train):
    print y_train
    y_train_hot = to_categorical(y_train, 3)
    print y_train_hot

    ## Create your own model here given the constraints in the problem
    model = Sequential()
    model.add(Dense(1000, input_dim=len(X_train[0])))  # Use np.reshape instead of this in hw
    model.add(Activation('linear'))

    model.add(Dense(50))
    model.add(Activation('relu'))

    ## Once you one-hot encode the data labels, the line below should be predicting probabilities of each of the 10 classes
    ## e.g. it should read: model.add(Dense(10)), not model.add(Dense(1))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    ## Printing a summary of the layers and weights in your model
    model.summary()

    ## In the line below we have specified the loss function as 'mse' (Mean Squared Error) because in the above code we did not one-hot encode the labels.
    ## In your implementation, since you are one-hot encoding the labels, you should use 'categorical_crossentropy' as your loss.
    ## You will likely have the best results with RMS prop or Adam as your optimizer.  In the line below we use Adadelta
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

    fit = model.fit(X_train, y_train_hot, batch_size=32, nb_epoch=10,
        verbose=1)

    return model



def compute_error_clf(clf, x, y):
    # Returns fraction of misclassified points
    pred = clf.predict(x)
    res = [1 if pred[i] != y[i] else 0 for i in range(len(pred))]
    return float(sum(res)) / len(res)


def get_feature_set():
    s = 'id,HRMONTH,HRYEAR4,HURESPLI,HUFINAL,HUSPNISH,HETENURE,HEHOUSUT,HETELHHD,HETELAVL,HEPHONEO,HUFAMINC,HUTYPEA,HUTYPB,HUTYPC,HWHHWGT,HRINTSTA,HRNUMHOU,HRHTYPE,HRMIS,HUINTTYP,HUPRSCNT,HRLONGLK,HRHHID2,HUBUS,HUBUSL1,HUBUSL2,HUBUSL3,HUBUSL4,GEREG,GESTCEN,GESTFIPS,GTCBSA,GTCO,GTCBSAST,GTMETSTA,GTINDVPC,GTCBSASZ,GTCSA,PERRP,PEPARENT,PEAGE,PRTFAGE,PEMARITL,PESPOUSE,PESEX,PEAFEVER,PEAFNOW,PEEDUCA,PTDTRACE,PRDTHSP,PUCHINHH,PULINENO,PRFAMNUM,PRFAMREL,PRFAMTYP,PEHSPNON,PRMARSTA,PRPERTYP,PENATVTY,PEMNTVTY,PEFNTVTY,PRCITSHP,PRCITFLG,PRINUSYR,PUSLFPRX,PEMLR,PUWK,PUBUS1,PUBUS2OT,PUBUSCK1,PUBUSCK2,PUBUSCK3,PUBUSCK4,PURETOT,PUDIS,PERET1,PUDIS1,PUDIS2,PUABSOT,PULAY,PEABSRSN,PEABSPDO,PEMJOT,PEMJNUM,PEHRUSL1,PEHRUSL2,PEHRFTPT,PEHRUSLT,PEHRWANT,PEHRRSN1,PEHRRSN2,PEHRRSN3,PUHROFF1,PUHROFF2,PUHROT1,PUHROT2,PEHRACT1,PEHRACT2,PEHRACTT,PEHRAVL,PUHRCK1,PUHRCK2,PUHRCK3,PUHRCK4,PUHRCK5,PUHRCK6,PUHRCK7,PUHRCK12,PULAYDT,PULAY6M,PELAYAVL,PULAYAVR,PELAYLK,PELAYDUR,PELAYFTO,PULAYCK1,PULAYCK2,PULAYCK3,PULK,PELKM1,PULKM2,PULKM3,PULKM4,PULKM5,PULKM6,PULKDK1,PULKDK2,PULKDK3,PULKDK4,PULKDK5,PULKDK6,PULKPS1,PULKPS2,PULKPS3,PULKPS4,PULKPS5,PULKPS6,PELKAVL,PULKAVR,PELKLL1O,PELKLL2O,PELKLWO,PELKDUR,PELKFTO,PEDWWNTO,PEDWRSN,PEDWLKO,PEDWWK,PEDW4WK,PEDWLKWK,PEDWAVL,PEDWAVR,PUDWCK1,PUDWCK2,PUDWCK3,PUDWCK4,PUDWCK5,PEJHWKO,PUJHDP1O,PEJHRSN,PEJHWANT,PUJHCK1,PUJHCK2,PRABSREA,PRCIVLF,PRDISC,PREMPHRS,PREMPNOT,PREXPLF,PRFTLF,PRHRUSL,PRJOBSEA,PRPTHRS,PRPTREA,PRUNEDUR,PRUNTYPE,PRWKSCH,PRWKSTAT,PRWNTJOB,PUJHCK3,PUJHCK4,PUJHCK5,PUIODP1,PUIODP2,PUIODP3,PEIO1COW,PUIO1MFG,PEIO2COW,PUIO2MFG,PUIOCK1,PUIOCK2,PUIOCK3,PRIOELG,PRAGNA,PRCOW1,PRCOW2,PRCOWPG,PRDTCOW1,PRDTCOW2,PRDTIND1,PRDTIND2,PRDTOCC1,PRDTOCC2,PREMP,PRMJIND1,PRMJIND2,PRMJOCC1,PRMJOCC2,PRMJOCGR,PRNAGPWS,PRNAGWS,PRSJMJ,PRERELG,PEERNUOT,PEERNPER,PEERNRT,PEERNHRY,PUERNH1C,PEERNH2,PEERNH1O,PRERNHLY,PTHR,PEERNHRO,PRERNWA,PTWK,PEERN,PUERN2,PTOT,PEERNWKP,PEERNLAB,PEERNCOV,PENLFJH,PENLFRET,PENLFACT,PUNLFCK1,PUNLFCK2,PESCHENR,PESCHFT,PESCHLVL,PRNLFSCH,PWFMWGT,PWLGWGT,PWORWGT,PWSSWGT,PWVETWGT,PRCHLD,PRNMCHLD,PRWERNAL,PRHERNAL,HXTENURE,HXHOUSUT,HXTELHHD,HXTELAVL,HXPHONEO,PXINUSYR,PXRRP,PXPARENT,PXAGE,PXMARITL,PXSPOUSE,PXSEX,PXAFWHN1,PXAFNOW,PXEDUCA,PXRACE1,PXNATVTY,PXMNTVTY,PXFNTVTY,PXHSPNON,PXMLR,PXRET1,PXABSRSN,PXABSPDO,PXMJOT,PXMJNUM,PXHRUSL1,PXHRUSL2,PXHRFTPT,PXHRUSLT,PXHRWANT,PXHRRSN1,PXHRRSN2,PXHRACT1,PXHRACT2,PXHRACTT,PXHRRSN3,PXHRAVL,PXLAYAVL,PXLAYLK,PXLAYDUR,PXLAYFTO,PXLKM1,PXLKAVL,PXLKLL1O,PXLKLL2O,PXLKLWO,PXLKDUR,PXLKFTO,PXDWWNTO,PXDWRSN,PXDWLKO,PXDWWK,PXDW4WK,PXDWLKWK,PXDWAVL,PXDWAVR,PXJHWKO,PXJHRSN,PXJHWANT,PXIO1COW,PXIO1ICD,PXIO1OCD,PXIO2COW,PXIO2ICD,PXIO2OCD,PXERNUOT,PXERNPER,PXERNH1O,PXERNHRO,PXERN,PXERNWKP,PXERNRT,PXERNHRY,PXERNH2,PXERNLAB,PXERNCOV,PXNLFJH,PXNLFRET,PXNLFACT,PXSCHENR,PXSCHFT,PXSCHLVL,QSTNUM,OCCURNUM,PEDIPGED,PEHGCOMP,PECYC,PEGRPROF,PEGR6COR,PEMS123,PXDIPGED,PXHGCOMP,PXCYC,PXGRPROF,PXGR6COR,PXMS123,PWCMPWGT,PEIO1ICD,PEIO1OCD,PEIO2ICD,PEIO2OCD,PRIMIND1,PRIMIND2,PEAFWHN1,PEAFWHN2,PEAFWHN3,PEAFWHN4,PXAFEVER,PELNDAD,PELNMOM,PEDADTYP,PEMOMTYP,PECOHAB,PXLNDAD,PXLNMOM,PXDADTYP,PXMOMTYP,PXCOHAB,PEDISEAR,PEDISEYE,PEDISREM,PEDISPHY,PEDISDRS,PEDISOUT,PRDISFLG,PXDISEAR,PXDISEYE,PXDISREM,PXDISPHY,PXDISDRS,PXDISOUT,PES1'
    ids = s.split(',')

    feature_lst = ['HETENURE', 'HEHOUSUT', 'HETELHHD','HEPHONEO','HUFAMINC','HWHHWGT','HRNUMHOU','HRHTYPE','HRMIS','HUINTTYP','HUPRSCNT','HUBUS','GEREG','GESTCEN','GESTFIPS','GTCBSAST','GTCBSASZ','PERRP', 'PEMARITL','PEAGE','PESEX','PEAFEVER','PEEDUCA','PTDTRACE','PRDTHSP','PEHSPNON','PENATVTY','PRCITSHP','PURETOT','PUDIS','PEHRUSL1','PRCIVLF','PREMPHRS','PEIO1COW','PUIO1MFG','PRDTIND1','PRMJOCGR','PRSJMJ','PRCHLD',]
    feature_set = [ids.index(x) for x in feature_lst]

    # feature_set = range(len(s.split(',')))[1:-1]
    return feature_set


def extract_feature(columns, data):
    col_data = [data[:, i] for i in columns]
    filter_data = np.asarray(zip(*col_data))
    assert len(filter_data) == len(data)
    return filter_data


def process_test_data(file_name, s=','):
    f = open(file_name, 'r')
    f.readline()
    data = np.asarray([[float(j) for j in i.split(s)] for i in f])
    return np.asarray(data)

def process_train_data(file_name, s=','):
    f = open(file_name, 'r')
    f.readline()
    data = np.asarray([[float(j) for j in i.split(s)] for i in f])
    x, y = zip(*[(i[:-1], i[-1]) for i in data])
    return np.asarray(x), np.asarray(y)

def gen_file(pred):
    f = open('kaggle_submission.csv', 'w')
    header = 'id,PES1'
    f.write(header + '\n')
    for i in range(16000):
        f.write(str(i) + ',' + str(int(pred[i])) + '\n')
    f.close()

kaggle_classify()