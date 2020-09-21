''' submodulo que faz os graficos do hsp'''
## cria uma imagem baseado na intensidade  

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
    
## cria uma imagem baseado alpha de uma região entre ini1 e fim1 alpha e definido pela eq spci = alpha*meanspc     
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
# plota nspc espectros aleatorios 
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
 # plota uma imagem na qual o intensidade do pixel é um coeff do emsc   
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
    
## cria uma imagem baseado na intensidade e permite a vizualização do espectro de cada pixel
def int_plt(dados,wnsel):
    import numpy as np
    import matplotlib.pyplot as plt
    r = dados['r'].copy()
    dx = dados['dx']
    dy = dados['dy']
    sel = dados['sel']
    wn = dados['wn']
    rr = np.zeros((sel.shape[0],r.shape[1]))
    rr[sel,:] = r
    sel = wn>wnsel
    r = 0
    rr = rr.reshape(dx,dy,-1)
    plt.close('all')
    plt.figure(1)
    z = np.arange(0,rr.shape[2])
    z = z[sel][0]
    plt.pcolor(rr[:,:,z])
    x = 15
    y = 15
    while (x > 14 or y > 14 ):
        ver = plt.ginput(1)
        x = int(ver[0][0])
        y = int(ver[0][1])
        print(x,'  ',y)
        plt.close(2)
        plt.figure(2)
        plt.plot(wn,rr[y,x,:])
        plt.title('espectro do pixel x='+str(x)+' y= '+ str(y))
        plt.figure(1)
    plt.close(1)
    plt.close(2)


## cria uma imagem baseado na area entre a e b e permite a vizualização do espectro de cada pixel     
def area_plt(dados,a,b):
    import numpy as np
    import matplotlib.pyplot as plt
    
    sel = (dados['wn'] > (a)) &  (dados['wn'] < (b) )
    r = dados['r'][:,sel]
    area = np.trapz(r)
    print(area.min)
    dplot = np.zeros(dados['dx']*dados['dy']);
    dplot[dados['sel']] = area
    dplot =dplot.reshape(dados['dx'],dados['dy'])
    plt.figure(1)
    plt.pcolor(dplot, vmin=np.min(area), vmax=np.max(area))
    plt.clim(np.min(area),np.max(area))
    plt.colorbar()
    l = 'imagem da area da banda entre ' +str(a)+ ' cm-1'+str(b)+ ' cm-1 \n ' + str(dados['filename'])[:-4]
    plt.title(l)  
    
    r = dados['r'].copy()
    dx = dados['dx']
    dy = dados['dy']
    sel = dados['sel']
    wn = dados['wn']
    rr = np.zeros((sel.shape[0],r.shape[1]))
    rr[sel,:] = r
    rr = rr.reshape(dx,dy,-1)
    x = 15
    y = 15
    while (x > 14 or y > 14 ):
        ver = plt.ginput(1)
        x = int(ver[0][0])
        y = int(ver[0][1])
        print(x,'  ',y)
        plt.close(2)
        plt.figure(2)
        plt.plot(wn,rr[y,x,:])
        plt.title('espectro do pixel x='+str(x)+' y= '+ str(y))
        plt.figure(1)
    plt.close(1)
    plt.close(2)