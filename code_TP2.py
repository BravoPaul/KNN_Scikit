from time import clock
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

DonneeTrain = np.load('E:\\dossieer\\cours_BigData\\chiffres\\data\\trn_img.npy')
DonneeTrainCor = np.load('E:\\dossieer\\cours_BigData\\chiffres\\data\\trn_lbl.npy')
DonneeTest = np.load('E:\\dossieer\\cours_BigData\\chiffres\\data\\dev_img.npy')
DonneeCorrect = np.load('E:\\dossieer\\cours_BigData\\chiffres\\data\\dev_lbl.npy')

def mesureTest(train = DonneeTrain,trainLabel = DonneeTrainCor,trainTest = DonneeTest,trainTestLabel = DonneeCorrect) :
	for algo in ['kd_tree', 'ball_tree']:
		print("algo est:",algo)
		for mesureP in ['euclidean','manhattan','chebyshev','minkowski']:
			print("Mesure de distance est",mesureP)
			time1 = clock()
			neigh = KNeighborsClassifier(n_neighbors=1, algorithm=algo,metric = mesureP,p=2, n_jobs=-1)
			neigh.fit(train, trainLabel)
			PredireLabel = neigh.predict(trainTest)
			time3 = clock()
			print("time in total:",(time3-time1))
			TauxErreur = np.count_nonzero(PredireLabel != trainTestLabel)/float(len(trainTestLabel))
			print("Taux d'erreur est:",TauxErreur)

def preTraiteMesure():
	for compo in [20,25,30,35,40,45,50,55,60,70]:
		print("on utilise la dimention ",compo)
		pca = PCA(n_components=compo)
		tr_donneTrain = pca.fit(DonneeTrain).transform(DonneeTrain)
		tr_donneTest = pca.fit(DonneeTrain).transform(DonneeTest)
		time1 = clock()
		nbrs = KNeighborsClassifier(n_neighbors=4,weights='distance',algorithm = 'ball_tree',metric='euclidean').fit(tr_donneTrain,DonneeTrainCor)
		PredireLabel = nbrs.predict(tr_donneTest)
		time3 = clock()
		print("time in total:",(time3-time1))
		TauxErreur = np.count_nonzero(PredireLabel != DonneeCorrect)/float(len(DonneeCorrect))
		print("Taux d'erreur est:",TauxErreur)

def pondereTest():
	for algo in ['kd_tree', 'ball_tree']:
		print("algo est:",algo)
		for nombreNeigh in [1,2,3,4,5]:	
			print("nombre de voisin:",nombreNeigh)
			for weight in ['uniform','distance']:
				print("weights est",weight)
				time1 = clock()
				nbrs = KNeighborsClassifier(n_neighbors=nombreNeigh,weights=weight,algorithm = algo,metric='euclidean').fit(DonneeTrain,DonneeTrainCor)
				PredireLabel = nbrs.predict(DonneeTest)
				time3 = clock()
				print("time in total:",(time3-time1))
				TauxErreur = np.count_nonzero(PredireLabel != DonneeCorrect)/float(len(DonneeCorrect))
				print("Taux d'erreur est:",TauxErreur)

def prodMatFusion():
	pca = PCA(n_components=55)
	tr_donneTrain = pca.fit(DonneeTrain).transform(DonneeTrain)
	tr_donneTest = pca.fit(DonneeTrain).transform(DonneeTest)
	nbrs = KNeighborsClassifier(n_neighbors=4,weights='distance',algorithm = 'ball_tree',metric='euclidean').fit(tr_donneTrain,DonneeTrainCor)
	PredireLabel = nbrs.predict(tr_donneTest)
	matrice = np.zeros(shape = (10,10))
	for indice in range(len(DonneeCorrect)):
		matrice[DonneeCorrect[indice]][PredireLabel[indice]] = matrice[DonneeCorrect[indice]][PredireLabel[indice]]+1
	np.save('MatriceFusion',matrice)


##mesureTest()
##pondereTest()
##preTraiteMesure()
prodMatFusion()