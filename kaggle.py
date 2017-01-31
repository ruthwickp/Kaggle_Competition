import random
import numpy as np
from sklearn import svm
import pickle


def kaggle_classify():
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

    # Get the feature set 
    feature_set = get_feature_set()

    print 'Feature set: ', feature_set
    filter_train_x = extract_feature(feature_set, train_x)
    filter_train_y = train_y
    print 'Finished filtering...'

    # clf = svm.SVC(C=10**10, gamma=10**-20)
    # clf = svm.SVC(kernel='sigmoid', C=10**20, gamma=10**-16)
    clf = svm.SVC(C=float('inf'), gamma=10**-17, max_iter=10**7)
    # clf = svm.SVC(C=10**20, gamma=10**-15)
    # clf = svm.SVC(C=float('inf'), gamma=10**-15)
    clf.fit(filter_train_x[::50], filter_train_y[::50])
    print 'Finished classfying...'

    # Compute error
    valid_step = 1000
    avg = []
    for i in range(len(filter_train_x) / valid_step):
        start, stop = valid_step * i, valid_step * (i + 1)
        err = compute_error_clf(clf, filter_train_x[start:stop], filter_train_y[start:stop])
        print 'Validation error from (%d, %d): %g' % (start, stop, err)
        avg.append(err)

    avg_err = float(sum(avg)) / len(avg)
    print 'Average err: ', avg_err

    test_x = process_test_data('test_2008.csv')
    filter_test_x = extract_feature(feature_set, test_x)
    print len(filter_test_x)
    pred = clf.predict(filter_test_x)
    print pred
    gen_file(pred)


def compute_error_clf(clf, x, y):
    # Returns fraction of misclassified points
    pred = clf.predict(x)
    res = [1 if pred[i] == y[i] else 0 for i in range(len(pred))]
    return float(sum(res)) / len(res)


def get_feature_set():
    s = 'id,HRMONTH,HRYEAR4,HURESPLI,HUFINAL,HUSPNISH,HETENURE,HEHOUSUT,HETELHHD,HETELAVL,HEPHONEO,HUFAMINC,HUTYPEA,HUTYPB,HUTYPC,HWHHWGT,HRINTSTA,HRNUMHOU,HRHTYPE,HRMIS,HUINTTYP,HUPRSCNT,HRLONGLK,HRHHID2,HUBUS,HUBUSL1,HUBUSL2,HUBUSL3,HUBUSL4,GEREG,GESTCEN,GESTFIPS,GTCBSA,GTCO,GTCBSAST,GTMETSTA,GTINDVPC,GTCBSASZ,GTCSA,PERRP,PEPARENT,PEAGE,PRTFAGE,PEMARITL,PESPOUSE,PESEX,PEAFEVER,PEAFNOW,PEEDUCA,PTDTRACE,PRDTHSP,PUCHINHH,PULINENO,PRFAMNUM,PRFAMREL,PRFAMTYP,PEHSPNON,PRMARSTA,PRPERTYP,PENATVTY,PEMNTVTY,PEFNTVTY,PRCITSHP,PRCITFLG,PRINUSYR,PUSLFPRX,PEMLR,PUWK,PUBUS1,PUBUS2OT,PUBUSCK1,PUBUSCK2,PUBUSCK3,PUBUSCK4,PURETOT,PUDIS,PERET1,PUDIS1,PUDIS2,PUABSOT,PULAY,PEABSRSN,PEABSPDO,PEMJOT,PEMJNUM,PEHRUSL1,PEHRUSL2,PEHRFTPT,PEHRUSLT,PEHRWANT,PEHRRSN1,PEHRRSN2,PEHRRSN3,PUHROFF1,PUHROFF2,PUHROT1,PUHROT2,PEHRACT1,PEHRACT2,PEHRACTT,PEHRAVL,PUHRCK1,PUHRCK2,PUHRCK3,PUHRCK4,PUHRCK5,PUHRCK6,PUHRCK7,PUHRCK12,PULAYDT,PULAY6M,PELAYAVL,PULAYAVR,PELAYLK,PELAYDUR,PELAYFTO,PULAYCK1,PULAYCK2,PULAYCK3,PULK,PELKM1,PULKM2,PULKM3,PULKM4,PULKM5,PULKM6,PULKDK1,PULKDK2,PULKDK3,PULKDK4,PULKDK5,PULKDK6,PULKPS1,PULKPS2,PULKPS3,PULKPS4,PULKPS5,PULKPS6,PELKAVL,PULKAVR,PELKLL1O,PELKLL2O,PELKLWO,PELKDUR,PELKFTO,PEDWWNTO,PEDWRSN,PEDWLKO,PEDWWK,PEDW4WK,PEDWLKWK,PEDWAVL,PEDWAVR,PUDWCK1,PUDWCK2,PUDWCK3,PUDWCK4,PUDWCK5,PEJHWKO,PUJHDP1O,PEJHRSN,PEJHWANT,PUJHCK1,PUJHCK2,PRABSREA,PRCIVLF,PRDISC,PREMPHRS,PREMPNOT,PREXPLF,PRFTLF,PRHRUSL,PRJOBSEA,PRPTHRS,PRPTREA,PRUNEDUR,PRUNTYPE,PRWKSCH,PRWKSTAT,PRWNTJOB,PUJHCK3,PUJHCK4,PUJHCK5,PUIODP1,PUIODP2,PUIODP3,PEIO1COW,PUIO1MFG,PEIO2COW,PUIO2MFG,PUIOCK1,PUIOCK2,PUIOCK3,PRIOELG,PRAGNA,PRCOW1,PRCOW2,PRCOWPG,PRDTCOW1,PRDTCOW2,PRDTIND1,PRDTIND2,PRDTOCC1,PRDTOCC2,PREMP,PRMJIND1,PRMJIND2,PRMJOCC1,PRMJOCC2,PRMJOCGR,PRNAGPWS,PRNAGWS,PRSJMJ,PRERELG,PEERNUOT,PEERNPER,PEERNRT,PEERNHRY,PUERNH1C,PEERNH2,PEERNH1O,PRERNHLY,PTHR,PEERNHRO,PRERNWA,PTWK,PEERN,PUERN2,PTOT,PEERNWKP,PEERNLAB,PEERNCOV,PENLFJH,PENLFRET,PENLFACT,PUNLFCK1,PUNLFCK2,PESCHENR,PESCHFT,PESCHLVL,PRNLFSCH,PWFMWGT,PWLGWGT,PWORWGT,PWSSWGT,PWVETWGT,PRCHLD,PRNMCHLD,PRWERNAL,PRHERNAL,HXTENURE,HXHOUSUT,HXTELHHD,HXTELAVL,HXPHONEO,PXINUSYR,PXRRP,PXPARENT,PXAGE,PXMARITL,PXSPOUSE,PXSEX,PXAFWHN1,PXAFNOW,PXEDUCA,PXRACE1,PXNATVTY,PXMNTVTY,PXFNTVTY,PXHSPNON,PXMLR,PXRET1,PXABSRSN,PXABSPDO,PXMJOT,PXMJNUM,PXHRUSL1,PXHRUSL2,PXHRFTPT,PXHRUSLT,PXHRWANT,PXHRRSN1,PXHRRSN2,PXHRACT1,PXHRACT2,PXHRACTT,PXHRRSN3,PXHRAVL,PXLAYAVL,PXLAYLK,PXLAYDUR,PXLAYFTO,PXLKM1,PXLKAVL,PXLKLL1O,PXLKLL2O,PXLKLWO,PXLKDUR,PXLKFTO,PXDWWNTO,PXDWRSN,PXDWLKO,PXDWWK,PXDW4WK,PXDWLKWK,PXDWAVL,PXDWAVR,PXJHWKO,PXJHRSN,PXJHWANT,PXIO1COW,PXIO1ICD,PXIO1OCD,PXIO2COW,PXIO2ICD,PXIO2OCD,PXERNUOT,PXERNPER,PXERNH1O,PXERNHRO,PXERN,PXERNWKP,PXERNRT,PXERNHRY,PXERNH2,PXERNLAB,PXERNCOV,PXNLFJH,PXNLFRET,PXNLFACT,PXSCHENR,PXSCHFT,PXSCHLVL,QSTNUM,OCCURNUM,PEDIPGED,PEHGCOMP,PECYC,PEGRPROF,PEGR6COR,PEMS123,PXDIPGED,PXHGCOMP,PXCYC,PXGRPROF,PXGR6COR,PXMS123,PWCMPWGT,PEIO1ICD,PEIO1OCD,PEIO2ICD,PEIO2OCD,PRIMIND1,PRIMIND2,PEAFWHN1,PEAFWHN2,PEAFWHN3,PEAFWHN4,PXAFEVER,PELNDAD,PELNMOM,PEDADTYP,PEMOMTYP,PECOHAB,PXLNDAD,PXLNMOM,PXDADTYP,PXMOMTYP,PXCOHAB,PEDISEAR,PEDISEYE,PEDISREM,PEDISPHY,PEDISDRS,PEDISOUT,PRDISFLG,PXDISEAR,PXDISEYE,PXDISREM,PXDISPHY,PXDISDRS,PXDISOUT,PES1'
    ids = s.split(',')

    feature_lst = ['HETENURE', 'HEHOUSUT', 'HETELHHD','HEPHONEO','HUFAMINC','HWHHWGT','HRNUMHOU','HRHTYPE','HRMIS','HUINTTYP','HUPRSCNT','HUBUS','GEREG','GESTCEN','GESTFIPS','GTCBSAST','GTCBSASZ','PERRP', 'PEMARITL','PEAGE','PESEX','PEAFEVER','PEEDUCA','PTDTRACE','PRDTHSP','PEHSPNON','PENATVTY','PRCITSHP','PURETOT','PUDIS','PEHRUSL1','PRCIVLF','PREMPHRS','PEIO1COW','PUIO1MFG','PRDTIND1','PRMJOCGR','PRSJMJ','PRCHLD',]
    feature_set = [ids.index(x) for x in feature_lst]
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