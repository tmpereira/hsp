''' submodulo que faz os graficos do hsp'''
 
def intt(datta,b):
    import matplotlib.pyplot as plt
    import numpy as np
    xx = 2
    yy = 2
    sel = datta ['wn'] > b
    ver = datta['r'][:,sel]
    ver = ver[:,0]
    dplot = np.zeros(datta['dx']*datta['dy']);
    dplot[datta['sel']] = ver
    dplot =dplot.reshape(datta['dx'],datta['dy'])
    plt.figure()
    plt.pcolor(dplot, vmin=np.min(ver), vmax=np.max(ver))
    plt.clim(np.min(ver),np.max(ver))
    plt.colorbar()
    l = 'imagem da intensidade ' +str(b)+ ' cm-1 \n ' + str(datta['filename'])[:-4]
    plt.title(l)
    plt.show()
    
    
## cria uma imagem baseado em area de uma banda 

def area(data,a,b):
    import numpy as np
    import matplotlib.pyplot as plt
    sel = (data['wn'] > (a)) &  (data['wn'] < (b) )
    r = data['r'][:,sel]
    area = np.trapz(r)
    print(area.min)
    dplot = np.zeros(data['dx']*data['dy']);
    dplot[data['sel']] = area
    dplot =dplot.reshape(data['dx'],data['dy'])
    plt.figure()
    plt.pcolor(dplot, vmin=np.min(area), vmax=np.max(area))
    plt.clim(np.min(area),np.max(area))
    plt.colorbar()
    l = 'imagem da area da banda entre ' +str(a)+ ' cm-1'+str(b)+ ' cm-1 \n ' + str(data['filename'])[:-4]
    plt.title(l)
    plt.show()
    
    
def mean(data,ini1,fim1):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import  LinearRegression
    sel = np.logical_and(data['wn'] > ini1,data['wn'] < fim1)
    r1 = data['r'][:,sel]
    media = np.mean(r1,axis=0).reshape(-1,1)
    meansvalue = np.zeros((data['r'].shape[0]))
    y = r1[:,:].T
    xx= np.vstack((r1.mean(axis=0),np.ones_like(r1[1,:]))).T
    alpha= np.linalg.lstsq(xx, y,rcond =-1)[0][0].T
    meansvalue = (alpha)   
    dplot = np.zeros(data['dx']*data['dy']);
    dplot[data['sel']] = meansvalue
    dplot =dplot.reshape(data['dx'],data['dy'])
    plt.pcolor(dplot,vmin=np.min(meansvalue), vmax=np.max(meansvalue))
    plt.clim(np.min(meansvalue),np.max(meansvalue))
    plt.colorbar()
    l = 'imagem da meanspc entre ' +str(ini1)+ ' cm-1'+str(fim1)+ ' cm-1 \n ' + str(data['filename'])[:-4]
    plt.title(l)
    return dplot
    
def pplot(data,nspc):
    import numpy as np
    import matplotlib.pyplot as plt
    r = data['r']
    k = np.random.randint(0,r.shape[0],(nspc),dtype='uint32')
    plt.figure()
    for i in k:
        plt.plot(data['wn'],r[i][:])
    plt.xlabel(' Número de onda')
    plt.show()
    
def emsc(datta,b):
    import matplotlib.pyplot as plt
    import numpy as np
    
    ver = datta['EMSC_coeff'][datta['sel'],b]
    dplot = np.zeros(datta['dx']*datta['dy']);
    dplot[datta['sel']] = ver
    dplot =dplot.reshape(datta['dx'],datta['dy'])
    plt.figure()
    plt.pcolor(dplot)
    plt.clim(np.min(ver),np.max(ver))
    plt.colorbar()
    
    l = 'histograma do coeficente EMSC' + str(datta['filename'])[:-4]
    plt.title(l)
    plt.show()