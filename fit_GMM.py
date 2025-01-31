import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
import argparse
import os
import pylab
import pickle
from astropy.io import fits
import sys

parser = argparse.ArgumentParser()

parser.add_argument(
                    '-f',
                    help="name of folder containing BEAGLE results",
                    action="store",
                    type=str,
                    dest="folder",
                    required=True
                    )

parser.add_argument(
                    '-p', '--plot-gmm',
                    help="make plots of GMM fits to posteriors",
                    action="store_true",
                    default=False,
                    dest="GMMplots",
                    required=False
                    )

parser.add_argument(
                    '--n-gauss',
                    help="how many Gaussians to fit to PDFs",
                    action="store",
                    type=int,
                    dest="nGauss",
                    required=True
                    )

parser.add_argument(
                    '--re-run',
                    help="if true then re-run fitting",
                    action="store_true",
                    default=False,
                    dest="reRun",
                    required=False
                    )

args = parser.parse_args()

os.system("mkdir "+args.folder+"GMM/")

if args.GMM is True:
    x = []
    y = []
    xycov = []
    xsig = []
    ysig = []
    amp = []
    if args.reRun:
        os.system('rm '+args.folder+"GMM/*.p")

    fileList = os.listdir(args.folder)
    for file in fileList:
        if '.fits.gz' in file:
            #check if the results for this individual object already stored
            pickleName = file.replace(".fits.gz","_GMM.p")
            fileTest = os.path.isfile(args.folder+"GMM/"+pickleName)
            if args.reRun or fileTest == False:
                print file
                data = fits.open(args.folder+file)
                probs_prop = np.array(data['POSTERIOR PDF'].data['probability'], np.float64)
                probs_prop = probs_prop/probs_prop.sum().astype(np.float64)
                sfr = np.log10(data['STAR FORMATION'].data['sfr'])
                mass = np.float64(data['GALAXY PROPERTIES'].data['M_star'])
                mass = np.log10(mass)
                idx = np.random.choice(len(probs_prop),size=1000, p=probs_prop)
                #gmm = pygmmis.GMM(K=1, D=2)      # K components, D dimensions
                gmm = GaussianMixture(n_components=args.nGauss, covariance_type='full')      # K components, D dimensions
                dataIn = np.array([mass[idx],sfr[idx]])

                tryCounter = 0
                fitted = False
                while fitted == False and tryCounter <= 3:
                    tryCounter = tryCounter+1
                    gmm.fit(dataIn)
                    try:
                        if args.nGauss > 1:
                            #Have to check that none of these give singular matrices!
                            minDet = 1E10
                            for j in range(args.nGauss):
                                cov = gmm.covar[j]
                                det = cov[0,0]*cov[1,1]-cov[0,1]*cov[1,0]
                                if det < minDet:
                                    minDet = det
            
                            if minDet > 1E-15:
                                fitted=True
                        else:
                            fitted=True
                    except:
                        print('failed ', tryCounter)
                            
                obs_size = 100
                samples= gmm.sample(obs_size)

                if args.GMMplots is True:
                    pylab.figure()
                    pylab.scatter(mass[idx],sfr[idx])
                    pylab.scatter(samples[:,0],samples[:,1],c='r')
                    plotName = file.replace(".fits.gz","_GMM_fit.pdf")
                    pylab.savefig(args.folder+"GMM/"+plotName)
                    pylab.close()
                    
                cov = gmm.covariances_
                temp_xsig = []
                temp_ysig = []
                temp_xycov = []
                for j in range(args.nGauss):
                    temp_xsig.append(np.sqrt(cov[j][0][0]))
                    temp_ysig.append(np.sqrt(cov[j][1][1]))
                    temp_xycov.append(cov[j][0][1])
                pickle.dump({'x':gmm.mean[:,0],'y':gmm.mean[:,1],'xsig':temp_xsig,'ysig':temp_ysig,'xycov':temp_xycov, 'amp':gmm.amp}, open(args.folder+"pygmmis_GMM_"+fileStr+"/"+pickleName,'w'))
            
