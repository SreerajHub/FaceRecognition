
# coding: utf-8

# In[1282]:


import cv2
import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from scipy import special
from scipy.optimize import minimize
from scipy.optimize import fmin

face_data="C:/Users/sreer/Desktop/Computer Vision ECE 763/srajend2_project1/0"
face_data_resized= "C:/Users/sreer/Desktop/Computer Vision ECE 763/srajend2_project1/resized_images"
back_data= "C:/Users/sreer/Desktop/Computer Vision ECE 763/srajend2_project1/background"
back_data_resized= "C:/Users/sreer/Desktop/Computer Vision ECE 763/srajend2_project1/background_resized"
test_data_resized = "C:/Users/sreer/Desktop/Computer Vision ECE 763/srajend2_project1/test_data_resized"


# In[1283]:


def resize_image(size,folder,outfolder):
    img_cnt = 0
    for filename in os.listdir(folder):
        if img_cnt<1000:
            img = Image.open(os.path.join(folder, filename))
            img=img.resize(size,Image.ANTIALIAS)
            img.save(os.path.join(outfolder, filename),"JPEG")
            img_cnt= img_cnt+1
    print("number of images resized: ", str(img_cnt))


# In[1284]:


def image_to_array(input_folder):
    images = []
    for filename in os.listdir(input_folder):
        img = cv2.imread(os.path.join(input_folder, filename), 0)
        img = img.flatten()
        if img is not None:
            images.append(img)
    images_arr = np.array(images)
    return(images_arr)


# In[1285]:


def get_test_list(input_folder):
    images = []
    for filename in os.listdir(input_folder):
        img = cv2.imread(os.path.join(input_folder, filename), 0)
        img = img.flatten()
        if img is not None:
            images.append(img)
    return(images)


# In[1286]:


def display(disp_array,title):
    im = np.array(disp_array, dtype=np.uint8)
    im.resize(20,20)
    plt.title(title)
    plt.imshow(im,cmap='gray')
    
    plt.show()


# In[1287]:


def frange(start, end, step):
    tmp = start
    while(tmp < end):
        yield tmp
        tmp += step  


# In[1288]:


def pdf(x,mu,cov):
    den = 1 / ( ((2* np.pi)**(len(mu)/2)) * (np.linalg.det(cov)**(1/2)) )
    num = (-1/2) * ((x-mu).T.dot(np.linalg.inv(cov))).dot((x-mu))
    return float(np.exp(num))
    #return float(den*np.exp(num))


# In[1289]:


def t_pdf(nu,mu,sig,x,D):
    f1=special.psi((nu+D)*0.5)
    #print("f1",f1)
    f2= ((x-mu).dot(np.linalg.inv(sig).dot((x-mu).T)))
    #print("f2",f2)
    f3= np.power((1+(f2/nu)),((nu+D)*(-0.5)))
    #print("f3",f3)
    f4= np.power((np.pi*nu),D/2)*np.sqrt(np.linalg.det(sig))*special.psi(nu*0.5)
    #print("f4",f4)
    L=(f1*f3)
    #print(L)
    
    return L
    


# In[1290]:


def plot(FP,TP):
    
    plt.plot(FP,TP, color='blue')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.show()
    return


# In[1291]:


def fit_cost(nu,p,q):
    
    n=nu/2
    I,D=p.shape
    val= I*(n*np.log(n))+(special.gammaln(n))
    val=val-(n-1)*np.sum(q)+n*np.sum(p)
    
    
    return val


# In[1292]:


def factor_prob(face_phi,nonface_phi,face_mean,face_cov,nonface_mean,nonface_cov,test_list,t):
    D,K=face_phi.shape
    test_result=[]
    num_face=0
    num_non_face=0
    face_cov= (face_phi.dot(face_phi.T))+face_cov
    nonface_cov=(nonface_phi.dot(nonface_phi.T))+nonface_cov
    for i in range(len(test_list)):
        
        pdf_face= np.exp(-0.5*((test_list[i]-face_mean).dot(np.linalg.inv(np.diagflat(np.diag(face_cov))).dot((test_list[i]-face_mean).T))))
        pdf_non_face=np.exp(-0.5*((test_list[i]-nonface_mean).dot(np.linalg.inv(np.diagflat(np.diag(nonface_cov))).dot((test_list[i]-nonface_mean).T))))
        prob_face= pdf_face/(pdf_face+pdf_non_face)
#         pdf_face=multivariate_normal.pdf(np.array(test_list[i]),face_mean,np.diagflat(np.diag(face_cov*np.power(10,100)))
#         pdf_non_face= multivariate_normal.pdf(np.array(test_list[i]),nonface_mean,np.diagflat(np.diag(nonface_cov*np.power(10,100)))
#         prob_face=pdf_face/(pdf_face+pdf_non_face)
        result=prob_face
        if prob_face>t:
            #result=1
            num_face=num_face+1
                                   
        else:
            #result=0
            num_non_face= num_non_face+1

            
        test_result.append(result)
        
    return(num_face,num_non_face)


# In[1293]:


def factor_em(x,K):
    #x- input array, K: number of factors
    I,D=x.shape
    #initialization:
    mu=np.mean(x,axis=0)
    phi=np.random.randn(D,K)
    sig=np.sum(np.square(x-mu),axis=0)/I
    f_iter=0
    
    while f_iter<1:
        #Expectation:
        inv_sig= np.diag(1./sig)
        var1=(phi.T).dot(inv_sig)
        var2=np.linalg.inv((var1.dot(phi))+ np.identity(K))
        E_hi=var2.dot((var1).dot((x-mu).T))
        E_hi_hi=[]
        for i in range(I):
            z=E_hi[:,i]
            temp=var2+ z.dot(z.T)
            E_hi_hi.append(temp)
                
        #Maximization
        phi1=np.zeros(shape=(D,K))
        
        
        
        
        for i in range(I):
            a=(E_hi[:,i]).T
            a=np.reshape(a,newshape=(1,K))
            x_mu=(x[i]-mu).T
            x_mu=np.reshape(x_mu,newshape=(D,1))
            phi1=phi1+(x_mu.dot(a))
        phi2=np.zeros(shape=(K,K))
        for i in range(I):
            phi2=phi2+E_hi_hi[i]
        phi2=np.linalg.inv(phi2)
        phi=phi1.dot(phi2)
        
        #Update sigma
        sig_dig=np.zeros(shape=(D,1))
        for i in range(I):
            xm=(x[i]-mu).T
            sig1=xm*xm
            
            sig2=(phi.dot(E_hi[:,i]))*xm
            sig_dig=sig_dig+sig1-sig2
        sig=sig_dig / I
        
             
        f_iter=f_iter+1
    #print("mu",mu)
    #print("phi",phi)
    #print("sig",sig)
    return(phi,mu,sig)
# if __name__=='__main__':
#     factor_em(face_array,2)
    


# In[1294]:


def factor(face_array,nonface_array,test_face_list,test_nonface_list,K):
        
    face_phi,face_mean,face_cov= factor_em(face_array,K)
    display(face_mean,'factor_face_mean')
    display(np.sqrt(np.diag(face_cov)),'factor_face_covariance')
    
    nonface_phi,nonface_mean,nonface_cov=factor_em(nonface_array,K)
    display(nonface_mean,'factor_NonFace_mean')
    display(np.sqrt(np.diag(nonface_cov)),'factor_NonFace_covariance')
    
    n_face_in_faces, n_nonface_in_faces= factor_prob(face_phi,nonface_phi,face_mean,face_cov,nonface_mean,nonface_cov,test_face_list,t=0.5)
    
    n_face_in_nonface,n_nonface_in_nonface= factor_prob(face_phi,nonface_phi,face_mean,face_cov,nonface_mean,nonface_cov,test_nonface_list,t=0.5)
    
    n_faces=len(test_face_list)
    n_nonfaces=len(test_nonface_list)
    FP=[]
    TP=[]
    false_positive_rate = np.divide(n_face_in_nonface, n_nonfaces)
    false_negative_rate = np.divide(n_nonface_in_faces,n_faces)
    missclassification_rate=np.divide((n_face_in_nonface+n_nonface_in_faces),(n_faces+n_nonfaces))
   
    print("number of faces detected in face(factor):", n_face_in_faces)
    print("number of non-faces detected in face data(factor):", n_nonface_in_faces)
    print("number of faces detected in non-face data (factor):", n_face_in_nonface)
    print("number of non-faces detected in non-face data(factor):", n_nonface_in_nonface)    
    print("false_positive_rate(factor)", n_face_in_nonface*0.01)
    print("false negative rate (factor):", n_nonface_in_faces*0.01)
    print("missclassification rate (factor)",(n_face_in_nonface+n_nonface_in_faces)*0.005)

    for j in frange(0,1,0.1):
        n_face_in_faces, n_nonface_in_faces= factor_prob(face_phi,nonface_phi,face_mean,face_cov,nonface_mean,nonface_cov,test_face_list,t=j)
        tp1=1-(n_nonface_in_faces*0.01)
        TP.append(tp1)
        n_face_in_nonface,n_nonface_in_nonface= factor_prob(face_phi,nonface_phi,face_mean,face_cov,nonface_mean,nonface_cov,test_nonface_list,t=j)
        fp1=n_face_in_nonface*0.01
        FP.append(fp1)
        
    fp=np.array(FP)
    tp=np.array(TP)
    print("false positive rate",fp)
    print("true positive rate",tp)
    plot(fp,tp)
    
    return


# In[1295]:



def em_mog(x,K):
    
    I,D=x.shape
    #initialise lamda
    lamda=np.full((K,1),(1./K)) 
    
    #initialise mu and sig
    mu=np.random.randint(100,255,size=(K,D))
    
    cov=[]
    for k in range(K):
        cov.append(10000*np.diag(np.diag(np.cov(x,rowvar=0))))
    
    mog_iter=0 
    while mog_iter<2:
        #Expectation
        l=np.zeros(shape=(I,K))
        r=np.zeros(shape=(I,K))
        sum_rik=0
        for i in range(I):
            sum_lik=0
            for k in range(K):
                
                l[i,k]=np.exp(-0.5*((x[i]-mu[k]).dot((np.linalg.inv(cov[k])).dot((x[i]-mu[k]).T))))
                sum_lik=sum_lik+l[i,k]
            for k in range(K):
                r[i,k]=l[i,k]/sum_lik
                sum_rik=sum_rik+r[i,k]
        #Maximization:
        
        cov=[]
        for k in range(K):
            lamda[k]=np.sum(r[:,k],axis=0)/sum_rik
            r1=r[:,k]
            rx=x * r1[:, np.newaxis]
            
            mu[k]=np.sum(rx,axis=0)/(np.sum(r[:,k]))
            
            
            sig=np.zeros(shape=(D,D))
            for i in range(I):
                num= r[i,k]*(((x[i]-mu[k]).T).dot(x[i]-mu[k]))
                
                sig=sig+num
            cov.append(np.diag(np.diag(sig/(np.sum(r[:,k])))))
            
        mog_iter=mog_iter+1
#     print("mog_lamda",lamda)
#     print("mog_mean",mu)
#     print("mog_covariance",cov)
    
    for k in range(K):
        display(mu[k],'mean')
        display(np.sqrt(np.diag(cov[k]/10000)),'covariance')
    
    return(lamda,mu,cov)
        
    
# if __name__ == '__main__':
#     mog(face_array,2)


# In[1297]:


def mog_prob(face_lamda,nonface_lamda,face_mean,face_cov,nonface_mean,nonface_cov, test_list,t):
    K,D=face_mean.shape
   
    num_face=0
    num_non_face=0
    # pdf_face=0
    # pdf_nonface=0
    for i in range(len(test_list)):
        pdf_face=0
        pdf_nonface=0
        for k in range(K):
            pdf_face=pdf_face+ face_lamda[k]*(np.exp(-0.5*((test_list[i]-face_mean[k]).dot(np.linalg.inv(np.diagflat(np.diag(face_cov[k]*(10**8))).dot((test_list[i]-face_mean[k]).T)))))
            pdf_nonface = pdf_nonface + nonface_lamda[k]*(np.exp(-0.5*((test_list[i]-nonface_mean[k]).dot(np.linalg.inv(np.diagflat(np.diag(nonface_cov[k]*(10**8)))).dot((test_list[i]-nonface_mean[k]).T)))))
        pdf_face=np.asscalar(np.array(pdf_face))
        pdf_nonface=np.asscalar(np.array(pdf_nonface))
        prob_face= pdf_face/(pdf_face+pdf_nonface)

        if prob_face>t:
            
            num_face=num_face+1
                                   
        else:
            #result=0
            num_non_face= num_non_face+1
    return(num_face,num_non_face)


# In[1296]:


def mog(face_array,nonface_array,test_face_list,test_nonface_list,K):
    face_lamda,face_mean,face_cov= em_mog(face_array,K)
    nonface_lamda,nonface_mean,nonface_cov=em_mog(nonface_array,K)
    
    n_face_in_faces, n_nonface_in_faces= mog_prob(face_lamda,nonface_lamda,face_mean,face_cov,nonface_mean,nonface_cov,test_face_list,t=0.5)
    n_face_in_nonface,n_nonface_in_nonface= mog_prob(face_lamda,nonface_lamda,face_mean,face_cov,nonface_mean,nonface_cov,test_nonface_list,t=0.5)
    n_faces=len(test_face_list)
    n_nonfaces=len(test_nonface_list)
    FP=[]
    TP=[]
   
    for j in frange(0,1,0.1):
        n_face_in_faces, n_nonface_in_faces= mog_prob(face_lamda,nonface_lamda,face_mean,face_cov,nonface_mean,nonface_cov,test_face_list,t=j)
        tp1=1-(n_nonface_in_faces*0.01)
        TP.append(tp1)
        n_face_in_nonface,n_nonface_in_nonface= mog_prob(face_lamda,nonface_lamda,face_mean,face_cov,nonface_mean,nonface_cov,test_nonface_list,t=j)
        fp1=n_face_in_nonface*0.01
        FP.append(fp1)
        
    fp=np.array(FP)
    tp=np.array(TP)
    print("false positive rate",fp)
    print("true positive rate",tp)
    plot(fp,tp)
    false_positive_rate = np.divide(n_face_in_nonface, n_nonfaces)
    false_negative_rate = np.divide(n_nonface_in_faces,n_faces)
    missclassification_rate=np.divide((n_face_in_nonface+n_nonface_in_faces),(n_faces+n_nonfaces))
   
    print("number of faces detected in face(mog):", n_face_in_faces)
    print("number of non-faces detected in face data(mog):", n_nonface_in_faces)
    print("number of faces detected in non-face data (mog):", n_face_in_nonface)
    print("number of non-faces detected in non-face data(mog):", n_nonface_in_nonface)    
    print("false_positive_rate", n_face_in_nonface*0.01)
    print("false negative rate:", n_nonface_in_faces*0.01)
    print("missclassification rate",(n_face_in_nonface+n_nonface_in_faces)*0.005)

    return
    


# In[1299]:


def em_mot(x,K):
    K=2
    I,D=x.shape
    pi=np.full((K,1),(1/K))
    mu1=np.random.randint(100,255,size=(1,D))
    mu2= np.random.randint(1,255,size=(1,D))
    sig1= np.diag(np.diag(np.cov(x,rowvar=0)))
    sig2=sig1
    nu1=10
    nu2=6
    t_iter=0
    L_prev=10000;
    delta1=delta2=np.zeros(shape=(I,1))
        
    while t_iter<1:
        #Expectation
        
        for i in range(I):
            delta1[i]= ((x[i]-mu1).dot(np.linalg.inv(sig1))).dot((x[i]-mu1).T)
            delta2[i]= ((x[i]-mu2).dot(np.linalg.inv(sig2))).dot((x[i]-mu2).T)
#        print(delta.shape)
        E_hi1=(nu1+D)/(nu1+delta1)
        E_hi2=(nu2+D)/(nu2+delta2)
        E_log_hi1=special.psi((nu1+D)/2)- np.log((nu1+delta1)/2)
        E_log_hi2=special.psi((nu2+D)/2)- np.log((nu2+delta2)/2)
        
        #Maximization
        E_hi_sum1=np.sum(E_hi1)
        E_hi_sum2=np.sum(E_hi2)
        mu1=(np.sum((E_hi1*x),axis=0))/E_hi_sum1
        mu2=(np.sum((E_hi2*x),axis=0))/E_hi_sum2
        sig1=np.zeros(shape=(D,D))
       
        sig2=np.zeros(shape=(D,D))
        
#         sig=((E_hi*((x-mu))).T).dot(x-mu)
        
        for i in range(I):
            temp1=x[i]-mu1
            temp2=x[i]-mu2
            temp1=np.reshape(temp1,(1,D))
            temp2=np.reshape(temp2,(1,D))
            sig1=sig1+E_hi1[i]*((temp1.T).dot(temp1))
            sig2=sig2+E_hi2[i]*((temp2.T).dot(temp2))
        
        sig1=sig1/E_hi_sum1
        sig2=sig2/E_hi_sum2
        
        sig1=np.diagflat(np.diag(sig1))
        sig2=np.diagflat(np.diag(sig2))
        #nu= minimize(fit_cost(nu,E_hi,E_log_hi),nu)
        nu1=minimize_cost(E_hi1,E_log_hi1)
        nu2=minimize_cost(E_hi2,E_log_hi2)
        
       
        t_iter=t_iter+1
#     #print("t_mean=",mu)
#     display(mu,'t_mean')
       # print("var",sig)
        #print("sqrt var",np.sqrt(np.diag(sig)))
#     display(((np.sqrt(np.diag(sig)))),'t_covariance')
        #print("df",nu)
    nu=[]
    mu=[]
    sig=[]
    nu.append(nu1)
    nu.append(nu2)
    mu.append(mu1)
    mu.append(mu2)
    sig.append(sig1)
    sig.append(sig2)

    return(pi,nu,mu,sig) 
    


# In[1298]:


def mot_prob(D,pi,face_nu,nonface_nu,face_mean,face_cov,nonface_mean,nonface_cov,test_list,t):
    test_result=[]
    K=len(pi)
    num_face=0
    num_non_face=0
    pdf_face=0
    pdf_non_face=0

    for i in range(len(test_list)):
        for k in range(K):
            
            pdf_face= pdf_face+ pi[k]*(t_pdf(face_nu[k],face_mean[k],np.diagflat(np.diag(face_cov[k]*np.power(10,30))),test_list[i],D))
            pdf_non_face= pdf_non_face + pi[k]*(t_pdf(nonface_nu[k],nonface_mean[k],np.diagflat(np.diag(nonface_cov[k]*np.power(10,30))),test_list[i],D))
#         pdf_face=np.asscalar(np.array(pdf_face))
#         pdf_non_face=np.asscalar(np.array(pdf_non_face))
        
            
        prob_face= float(pdf_face+(10**-20))/float(pdf_face+pdf_non_face+(10**-20))
        
        #result=prob_face
        if prob_face>t:
            num_face=num_face+1
        else:
            num_non_face= num_non_face+1
        #test_result.append(result)
        
    return(num_face,num_non_face)
    


# In[1300]:


def mot(face_array,nonface_array,test_face_list,test_nonface_list,K):
    I,D=face_array.shape    
    pi,face_nu,face_mean,face_cov= em_mot(face_array,K)
    pi,nonface_nu,nonface_mean,nonface_cov= em_mot(nonface_array,K)
    for k in range(K):
        display(face_mean[k],'mot_dist_face_mean')
        display(np.sqrt(np.diag(face_cov[k])),'mot_dist_face_covariance')


        display(nonface_mean[k],'mot_NonFace_mean')
        display(np.sqrt(np.diag(nonface_cov[k])),'mot_NonFace_covariance')
    
    n_face_in_faces, n_nonface_in_faces= mot_prob(D,pi,face_nu,nonface_nu,face_mean,face_cov,nonface_mean,nonface_cov,test_face_list,t=0.5)
    
    n_face_in_nonface,n_nonface_in_nonface= mot_prob(D,pi,face_nu,nonface_nu,face_mean,face_cov,nonface_mean,nonface_cov,test_nonface_list,t=0.5)
    
    n_faces=len(test_face_list)
    n_nonfaces=len(test_nonface_list)
    FP=[]
    TP=[]
    false_positive_rate = np.divide(n_face_in_nonface, n_nonfaces)
    false_negative_rate = np.divide(n_nonface_in_faces,n_faces)
    missclassification_rate=np.divide((n_face_in_nonface+n_nonface_in_faces),(n_faces+n_nonfaces))
   
    print("number of faces detected in face:", n_face_in_faces)
    print("number of non-faces detected in face data:", n_nonface_in_faces)
    print("number of faces detected in non-face data :", n_face_in_nonface)
    print("number of non-faces detected in non-face data:", n_nonface_in_nonface)    
    print("false_positive_rate", n_face_in_nonface*0.01)
    print("false negative rate :", n_nonface_in_faces*0.01)
    print("missclassification rate ",(n_face_in_nonface+n_nonface_in_faces)*0.005)

    for j in frange(0,1,0.1):
        n_face_in_faces, n_nonface_in_faces= mot_prob(D,pi,face_nu,nonface_nu,face_mean,face_cov,nonface_mean,nonface_cov,test_face_list,t=j)

        tp1=1-(n_nonface_in_faces*0.01)
        TP.append(tp1)
        n_face_in_nonface,n_nonface_in_nonface= mot_prob(D,pi,face_nu,nonface_nu,face_mean,face_cov,nonface_mean,nonface_cov,test_nonface_list,t=j)
        fp1=n_face_in_nonface*0.01
        FP.append(fp1)
        
    fp=np.array(FP)
    tp=np.array(TP)
    print("false positive rate",fp)
    print("true positive rate",tp)
    plot(fp,tp)
    
    return


# In[1301]:


def t_dist_em(x):
    I,D=x.shape
    mu=np.mean(x,axis=0)
    #print(mu.shape)
    #sig=np.cov(x,rowvar=0)
    sig=np.sum(np.square(x-mu),axis=0)/I
    sig=np.diagflat(sig)
    #print(sig)
    nu=1000
    t_iter=0
    L_prev=10000;
    delta=np.zeros(shape=(I,1))
    

    while t_iter<1:
        #Expectation
        for i in range(I):
            delta[i]= ((x[i]-mu).dot(np.linalg.inv(sig))).dot((x[i]-mu).T)
#        print(delta.shape)
        E_hi=(nu+D)/(nu+delta)
        E_log_hi=special.psi((nu+D)/2)- np.log((nu+delta)/2)
        
        #Maximization
        E_hi_sum=np.sum(E_hi)
        
        mu=(np.sum((E_hi*x),axis=0))/E_hi_sum
        sig=np.zeros(shape=(D,D))
        
#         sig=((E_hi*((x-mu))).T).dot(x-mu)
        
        for i in range(I):
            temp1=x[i]-mu
            temp1=np.reshape(temp1,(1,D))

            sig=sig+E_hi[i]*((temp1.T).dot(temp1))
        
        sig=sig/E_hi_sum
       # print(sig)
        sig=np.diagflat(np.diag(sig))

        #nu= minimize(fit_cost(nu,E_hi,E_log_hi),nu)
        nu=minimize_cost(E_hi,E_log_hi)
       
        t_iter=t_iter+1
#     #print("t_mean=",mu)
#     display(mu,'t_mean')
       # print("var",sig)
        #print("sqrt var",np.sqrt(np.diag(sig)))
#     display(((np.sqrt(np.diag(sig)))),'t_covariance')
        print("df",nu)


    return(nu,mu,sig)
# if __name__ == '__main__':
#     t_dist_em(face_array)


# In[1302]:


def minimize_cost(p,q):

    a=0
    d=1000
    i=0
    while i<50:
        t=(d-a)/3
        
        b=a+t
        c=d-t
        b_cost=fit_cost(b,p,q)
        #print("b",b_cost)
        c_cost=fit_cost(c,p,q)
        #print("c",c_cost)
        if b_cost < c_cost:
            d=c
        else:
            a=b
        i=i+1

#     if d-a<0:
#         nu=d
#     else:
#         nu=a
    
    return d


# In[1303]:


def t_prob(D,face_nu,nonface_nu,face_mean,face_cov,nonface_mean,nonface_cov,test_list,t):
  
    test_result=[]
    num_face=0
    num_non_face=0
    face_sig=np.diagflat(np.diag(face_cov*np.power(10,10)))
    #face_sig=np.diagflat(np.diag(face_cov))
    nonface_sig=np.diagflat(np.diag(nonface_cov*np.power(10,10)))
    #nonface_sig=nonface_cov*np.power(10,100)
    for i in range(len(test_list)):
        pdf_face= t_pdf(face_nu,face_mean,face_sig,test_list[i],D)
        pdf_non_face=t_pdf(nonface_nu,nonface_mean,nonface_sig,test_list[i],D)
        prob_face= pdf_face/(pdf_face+pdf_non_face)
        
        result=prob_face
        if prob_face>t:
            num_face=num_face+1
        else:
            num_non_face= num_non_face+1
        test_result.append(result)
        
    return(num_face,num_non_face)
        


# In[1304]:


def t_distribution(face_array,nonface_array,test_face_list,test_nonface_list):
    I,D=face_array.shape    
    face_nu,face_mean,face_cov= t_dist_em(face_array)
    display(face_mean,'t_dist_face_mean')
    display(np.sqrt(np.diag(face_cov)),'t_dist_face_covariance')
    
    nonface_nu,nonface_mean,nonface_cov= t_dist_em(nonface_array)
    display(nonface_mean,'t_NonFace_mean')
    display(np.sqrt(np.diag(nonface_cov)),'t_NonFace_covariance')
    
    n_face_in_faces, n_nonface_in_faces= t_prob(D,face_nu,nonface_nu,face_mean,face_cov,nonface_mean,nonface_cov,test_face_list,t=0.5)
    
    n_face_in_nonface,n_nonface_in_nonface= t_prob(D,face_nu,nonface_nu,face_mean,face_cov,nonface_mean,nonface_cov,test_nonface_list,t=0.5)
    
    n_faces=len(test_face_list)
    n_nonfaces=len(test_nonface_list)
    FP=[]
    TP=[]
    false_positive_rate = np.divide(n_face_in_nonface, n_nonfaces)
    false_negative_rate = np.divide(n_nonface_in_faces,n_faces)
    missclassification_rate=np.divide((n_face_in_nonface+n_nonface_in_faces),(n_faces+n_nonfaces))
   
    print("number of faces detected in face:", n_face_in_faces)
    print("number of non-faces detected in face data:", n_nonface_in_faces)
    print("number of faces detected in non-face data :", n_face_in_nonface)
    print("number of non-faces detected in non-face data:", n_nonface_in_nonface)    
    print("false_positive_rate", n_face_in_nonface*0.01)
    print("false negative rate :", n_nonface_in_faces*0.01)
    print("missclassification rate ",(n_face_in_nonface+n_nonface_in_faces)*0.005)

    for j in frange(0,1,0.1):
        n_face_in_faces, n_nonface_in_faces= t_prob(D,face_nu,nonface_nu,face_mean,face_cov,nonface_mean,nonface_cov,test_face_list,t=j)

        tp1=1-(n_nonface_in_faces*0.01)
        TP.append(tp1)
        n_face_in_nonface,n_nonface_in_nonface= t_prob(D,face_nu,nonface_nu,face_mean,face_cov,nonface_mean,nonface_cov,test_nonface_list,t=j)
        fp1=n_face_in_nonface*0.01
        FP.append(fp1)
        
    fp=np.array(FP)
    tp=np.array(TP)
    print("false positive rate",fp)
    print("true positive rate",tp)
    plot(fp,tp)
    
    return
    


# In[1305]:


def gaussian_prob(face_mean,face_cov,nonface_mean,nonface_cov, test_list,t):
    test_result=[]

    num_face=0
    num_non_face=0
    
    for i in range(len(test_list)):
        #pdf_g_face = np.exp(0.5 * np.matmul(np.matmul((test_list[i] - face_mean).T, np.linalg.inv(face_cov)), (test_list[i] - face_mean)))
        pdf_g_face= np.exp(-0.5*((test_list[i]-face_mean).dot(np.linalg.inv(np.diagflat(np.diag(face_cov))).dot((test_list[i]-face_mean).T))))
        #print("pdf_g_face shape=",pdf_g_face )
        #pdf_g_non_face= np.exp(0.5 * np.matmul(np.matmul((test_list[i] - nonface_mean).T, np.linalg.inv(nonface_cov)), (test_list[i] - nonface_mean)))
        pdf_g_non_face=np.exp(-0.5*((test_list[i]-nonface_mean).dot(np.linalg.inv(np.diagflat(np.diag(nonface_cov))).dot((test_list[i]-nonface_mean).T))))
        #pdf_g_face= multivariate_normal.pdf(test_list[i],face_mean,face_cov)
        #print("pdf_g_nonface =", pdf_g_non_face)
        #pdf_g_non_face = multivariate_normal.pdf(test_list[i],nonface_mean,nonface_cov)
        prob_face= pdf_g_face/(pdf_g_face+pdf_g_non_face)
            
        #print(prob_face)
        if prob_face>t:
            result=1
            num_face=num_face+1
        else:
            result=0
            num_non_face= num_non_face+1

        test_result.append(result)
    return(num_face,num_non_face)


# In[1306]:


def gaussian(face_array,nonface_array,test_face_list,test_nonface_list):
    face_mean = np.mean(face_array, axis=0)
    print("face mean shape:",face_mean.shape)
    face_cov = np.cov(face_array, rowvar=0)
    print("covariance shape:", face_cov.shape)
    display(face_mean,'face_mean')
    nonface_mean = np.mean(nonface_array, axis=0)
    print("Non-face mean shape:", nonface_mean.shape)
    nonface_cov = np.cov(nonface_array, rowvar=0)
    print("covariance shape:", nonface_cov.shape)
    display(nonface_mean,'nonface_mean')
    face_cov_disp=np.sqrt(np.diag(face_cov))
    nonface_cov_disp=np.sqrt(np.diag(nonface_cov))
    display(face_cov_disp,'face_covariance')
    display(nonface_cov_disp,'nonface_covariance')
    test_result1=[]
    test_result2=[]
    n_face_in_faces, n_nonface_in_faces= gaussian_prob(face_mean,face_cov,nonface_mean,nonface_cov,test_face_list,t=0.5)
    n_face_in_nonface,n_nonface_in_nonface= gaussian_prob(face_mean,face_cov,nonface_mean,nonface_cov,test_nonface_list,t=0.5)
    n_faces=len(test_face_list)
    n_nonfaces=len(test_nonface_list)
    false_positive_rate = np.divide(n_face_in_nonface, n_nonfaces)
    false_negative_rate = np.divide(n_nonface_in_faces,n_faces)
    missclassification_rate=np.divide((n_face_in_nonface+n_nonface_in_faces),(n_faces+n_nonfaces))
   
    print("number of faces detected in face(gaussian):", n_face_in_faces)
    print("number of non-faces detected in face data(gaussian):", n_nonface_in_faces)
    print("number of faces detected in non-face data(gaussian):", n_face_in_nonface)
    print("number of non-faces detected in non-face data(gaussian):", n_nonface_in_nonface)    
    print("false_positive_rate", n_face_in_nonface*0.01)
    print("false negative rate:", n_nonface_in_faces*0.01)
    print("missclassification rate",(n_face_in_nonface+n_nonface_in_faces)*0.005)

    
    FP=[]
    TP=[]
   
    for j in frange(0,1,0.1):
        n_face_in_faces, n_nonface_in_faces= gaussian_prob(face_mean,face_cov,nonface_mean,nonface_cov,test_face_list,t=j)
        tp1=1-(n_nonface_in_faces*0.01)
        TP.append(tp1)
        n_face_in_nonface,n_nonface_in_nonface= gaussian_prob(face_mean,face_cov,nonface_mean,nonface_cov,test_nonface_list,t=j)
        fp1=n_face_in_nonface*0.01
        FP.append(fp1)
        
    fp=np.array(FP)
    tp=np.array(TP)
    print(fp)
    print(tp)
    plot(fp,tp)
    
    return


# In[1308]:


import cv2
import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from scipy import special
from scipy.optimize import minimize
from scipy.optimize import fmin

face_data="C:/Users/sreer/Desktop/Computer Vision ECE 763/srajend2_project1/0"
face_data_resized= "C:/Users/sreer/Desktop/Computer Vision ECE 763/srajend2_project1/resized_images"
back_data= "C:/Users/sreer/Desktop/Computer Vision ECE 763/srajend2_project1/background"
back_data_resized= "C:/Users/sreer/Desktop/Computer Vision ECE 763/srajend2_project1/background_resized"
test_data_resized = "C:/Users/sreer/Desktop/Computer Vision ECE 763/srajend2_project1/test_data_resized"
testface_resized= "C:/Users/sreer/Desktop/Computer Vision ECE 763/srajend2_project1/testface_resized"
testnonface_resized="C:/Users/sreer/Desktop/Computer Vision ECE 763/srajend2_project1/testnonface_resized"

if __name__ == '__main__':

    #resize_image(size=(20,20),folder=face_data, outfolder=face_data_resized)
    #resize_image(size=(20,20), folder=back_data, outfolder=back_data_resized)
    face_array= image_to_array(face_data_resized)
    nonface_array= image_to_array(back_data_resized)
    test_list = get_test_list(test_data_resized)
    test_face_list=get_test_list(testface_resized)
    test_nonface_list=get_test_list(testnonface_resized)
    #print("shape of face array:", face_array.shape)
    #print("shape of non face array:",nonface_array.shape)
    #print("length of test_array: ", len(test_list))
    model=input("Enter model number: \n 1 for Gaussian \n 2 for Mixture of Gaussian \n 3 for t distribution \n 4 for mixture of t \n 5 for factor analyzer \n 6 for mixture of t factor analyzer \n 0 for all \n")
    if model==1:
        test_result_gaussian= gaussian(face_array, nonface_array,test_face_list,test_nonface_list)
        
    if model==2:
        K=input("number of mixtures: ")
        print("K:",K)
        mog(face_array,nonface_array,test_face_list,test_nonface_list,K)
    if model == 3:
        t_distribution(face_array,nonface_array,test_face_list,test_nonface_list)
    if model==4:
        K=input("K(default=2)")
        mot(face_array,nonface_array,test_face_list,test_nonface_list,K)
       
    if model==5:
        K=input("number of factors: ")
        print("K:",K)
        factor(face_array,nonface_array,test_face_list,test_nonface_list,K)
    if model==0:
        print("Gaussian in progress...")
        test_result_gaussian= gaussian(face_array, nonface_array,test_face_list,test_nonface_list)
        print("Mixture of Gaussian in progress...")
        K=input("number of mixtures: ")
        print("K:",K)
        mog(face_array,nonface_array,test_face_list,test_nonface_list,K)
        print("t-distribution in progress...")
        t_distribution(face_array,nonface_array,test_face_list,test_nonface_list)
        print("mixture of t...")
        K=input("K(default=2)")
        mot(face_array,nonface_array,test_face_list,test_nonface_list,K)
        print("factor analyzer...")
        K=input("number of factors: ")
        print("K:",K)
        factor(face_array,nonface_array,test_face_list,test_nonface_list,K)

