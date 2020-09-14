''' submodulo que faz histogramas para  definiçao dos qualit testes
'''

# faz o histogra da area do pixels entre a regiões a e b   
def area(data,ini,fim):
    import numpy as np
    import matplotlib.pyplot as plt
    sel = np.logical_and(data['wn'] > int(ini),data['wn'] < int(fim))
    r = data['r'][:,sel]
    area = np.trapz(r)  
    plt.figure()
    plt.hist(area,300)   
    
    linha = 'histograma de area entre '+str(ini)+ ' até ' + str(fim)
    plt.title(linha)
    plt.grid()

# faz o histograma da intensidade b dos pixels da imagem
    
def intt(data,b):
    import numpy as np
    import matplotlib.pyplot as plt
    sel = data ['wn'] > b
    ver = data['r'][:,sel]
    ver=ver[:,0]
    plt.figure()
    plt.hist(ver,300)
    linha = 'histograma da intensidade em '+ str(b) + ' cm-1'
    plt.title(linha)
    plt.grid()     

# faz o histograma da parametro do emsc 
    
def emsc(data,a):
    import numpy as np
    import matplotlib.pyplot as plt
    ver = data['EMSC_coeff'][:,a]
    plt.figure()
    plt.hist(ver,300)   
    linha = 'histograma so coeficiente '+ str(a) + '  do modelo de emsc'
    plt.title(linha)

def mean(data,ini1,fim1):
    import numpy as np
    import matplotlib.pyplot as plt
    sel = np.logical_and(data['wn'] > ini1,data['wn'] < fim1)
    r1 = data['r'][:,sel]
    media = np.mean(r1,axis=0).reshape(-1,1)
    meansvalue = np.zeros((data['r'].shape[0]))
    y = r1[:,:].T
    xx= np.vstack((r1.mean(axis=0),np.ones_like(r1[1,:]))).T
    alpha= np.linalg.lstsq(xx, y,rcond =-1)[0][0].T
    meansvalue = (alpha)   
    ver = (alpha)   
    plt.figure()
    plt.hist(ver,300)
    linha = 'histograma de area entre '+str(ini1)+ ' até ' + str(fim1)
    plt.title(linha)
    plt.grid()
    return ver      


