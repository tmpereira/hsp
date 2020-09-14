'''
submodulo para pre processamento das imagens de micro FTIR
'''

## recorta o especto entre a e b
        
def cut(data,a,b):
    import numpy as np
    sel1 = (data['wn'] > a )
    sel2 = (data['wn'] < b )
    ver = (sel1.astype(int) + sel2.astype(int))-1
    sel = ver.astype(bool)
    data['r'] = data['r'][:,sel]
    data['wn'] = data['wn'][sel]
# adicionando info ao log
    linha = '\n restricão espectral de ' +str(a) + ' cm-1 até ' + str(b) + ' cm-1'
    print(linha,end='')
    data['log'] = np.char.add(data['log'],linha)
    return data

# fazer normalização (SNV)
def snv(data):
    import numpy as np
    spc = data['r']
    media = np.mean(spc,axis=1)
    std = np.std(spc,axis=1)
    data['r'] = np.divide((spc - media[:,None]),std[:,None])
    # adicionando info ao log
    linha = '\n normalização SNV em unica região'
    print(linha,end='')
    data['log'] = np.char.add(data['log'],linha)
    return data

# fazer normalização vetorial (SNV)
def norm_vec(data):
    import numpy as np
    r = data['r']
    norma = (r*r)
    norma = np.sqrt(norma.sum(1)).reshape(-1,1)
    rnorm = np.tile(norma,(1,r.shape[1]))
    data['r'] = r/rnorm
    # adicionando info ao log
    linha = '\n normalização pela norma do vetor em unica região'
    print(linha,end='')
    data['log'] = np.char.add(data['log'],linha)
    return data

# filtro savitz golay

def golay(data,diff,order,win):
    import numpy as np
    from scipy.signal import savgol_coeffs
    from scipy.sparse import spdiags
    import numpy.matlib
    n = int((win-1)/2)
    sgcoeff = savgol_coeffs(win, order, deriv=diff)[:,None]
    sgcoeff = np.matlib.repmat(sgcoeff,1,data['r'].shape[1])
    diags = np.arange(-n,n+1)
    D = spdiags(sgcoeff,diags,data['r'].shape[1],data['r'].shape[1]).toarray()
    D[:,0:n] = 0
    D[:,data['r'].shape[1]-5:data['r'].shape[1]] = 0
    data['r'] = np.dot(data['r'],D)
 # adicionando info ao log
    linha = '\n filtro saviz golay usando \n'
    linha = linha + '  >> derivada ordem: ' + str(diff) +'\n'
    linha = linha + '  >> janela: ' + str(win) +'\n'
    linha = linha + '  >> polinomio: ' + str(order) + ' ordem'
    print(linha,end='')
    data['log'] = np.char.add(data['log'],linha)
    return data

 # normalizaçao em 2 regioes 
    
def norm2r(data,ini1,fim1,ini2,fim2):
    import numpy as np
    sel = np.logical_and(data['wn'] > int(ini1),data['wn'] < int(fim1))
    r1 = data['r'][:,sel]
    wn1 = data['wn'][sel][:,None]
    media = np.mean(r1,axis=1)
    std = np.std(r1,axis=1)
    r1 = np.divide((r1 - media[:,None]),std[:,None])
            
    sel = np.logical_and(data['wn'] > int(ini2),data['wn'] < int(fim2))
    r2 = data['r'][:,sel]
    wn2 = data['wn'][sel][:,None]
    media = np.mean(r2,axis=1)
    std = np.std(r2,axis=1)
    r2 = np.divide((r2 - media[:,None]),std[:,None])
    data['r'] = np.column_stack((r1,r2))
    data['wn'] = np.vstack((wn1,wn2))
    data['wn'] = data['wn'].reshape(-1)
    # adicionando info ao log
    linha = '\n normalização vetorial(SNV) em 2 regioes' + '\n'
    linha = linha + '  >> r1: '+str(ini1) + ' cm-1 até ' + str(fim1) + ' cm-1\n'
    linha = linha + '  >> r2: '+str(ini2) + ' cm-1 até ' + str(fim2) + ' cm-1'
    print(linha,end='')
    data['log'] = np.char.add(data['log'],linha)
    return data                

## remoção de ruido usando pca 
    
def pcares(data,n):
    import numpy as np
    from sklearn.decomposition import PCA
    print('inicializnado o pcares')
    pca = PCA()
    media = np.mean(data['r'],axis=0)
    pca.fit(data['r']-media)
    scoress = pca.transform(data['r'])
    scoress[:,n-1:-1] = 0 
    coeff= pca.components_
    data['r'] =media + np.dot(scoress,coeff)
     # adicionando info ao log
    linha = '\n remoção de ruido usando somente redução de PCA com ' +str(n) + ' pcs'
    print(linha,end='')
    data['log'] = np.char.add(data['log'],linha)
    return data

# funçao de faz denoising usando NAPC
def napc(dados,noise,npcs):
    d = dados['r']
    n = noise['r']
    import numpy as np
    print('inicializando NAPC')
    sigmad = np.cov(d.T)
    sigman = np.cov(n.T)
    [a,s1,e1] = np.linalg.svd(sigman)
    e1 = e1.T
    a = 1
    F = e1/np.sqrt(s1)
    
    sigma_adj = F.T @ sigmad @ F
    
    [a,b,G] = np.linalg.svd(sigma_adj)
    G = G.T
    
    H = F.dot(G)
    meanspc = (np.tile(d.mean(0).reshape(1,-1),(d.shape[0],1)))
    meandata =  d - meanspc
    
    
    scoresNAPC = H.T @ meandata.T
    scoresNAPC[npcs:,:] = 0
    
    zcorr = np.linalg.solve(H.T,scoresNAPC).T
    
    dados['r'] = zcorr + meanspc
    linha = '\n remoção de ruido usando somente redução de PCA com ' +str(n) + ' pcs'
    print(linha,end='')
    dados['log'] = np.char.add(dados['log'],linha)
    return dados

## remover offset nas regioes entre a e b
    
def offset(data,ini,fim):
    import numpy as np
    import matplotlib.pyplot as plt
    sel = np.logical_and(data['wn'] > int(ini),data['wn'] < int(fim));
    r = data['r'][:,sel];
    minino = np.min(r,axis=1);
    minino = np.reshape(minino,(-1,1));
    minino = np.tile(minino,data['r'].shape[1]);
    data['r'] = data['r']-minino;
    # adicionando info ao log
    linha = '\n remoção de offset usando o minimo valor entre ' +str(ini) + ' e ' + str(fim)
    print(linha,end='')
    data['log'] = np.char.add(data['log'],linha)
    return data

# binarizar imagem fazer cada 2X2 pixel se tornar pixel unico
    
def binned(data):
    import numpy as np
    import matplotlib.pyplot as plt    
    r = data['r']
    r = r.reshape(data['dx'],data['dy'],-1)
    dx = r.shape[0]
    dy = r.shape[1]
    dz = r.shape[2]
    dxbin = int(np.floor(dx/2))-1
    dybin = int(np.floor(dy/2))-1
    rbin = np.ones((dxbin,dybin,dz))
    jj = 0
    ii = 0
    for i in range(0,dy-2,2):
        for j in range(0,dx-2,2):
            sel = r[j:j+2,i:i+2,:];
            sel  = np.mean(sel.reshape(4,dz),axis=0)
            rbin[jj,ii,:] = sel
            jj = jj + 1
            
        jj = 0
        ii = ii + 1
    data['r'] = rbin.reshape((dxbin*dybin,dz)) 
    data['sel'] = np.ones((dxbin*dybin,)).astype('bool')
    data['dx'] = dxbin
    data['dy'] = dybin
    # adicionando info ao log
    linha = '\n dados binados e 2x2'
    print(linha,end='')
    data['log'] = np.char.add(data['log'],linha)
    return data

# seleçao aleatoria de k pixels
    
def rand(data,k):
    import numpy as np
    r = data['r']
    labels = np.random.permutation(r.shape[0])[:k]
    sel = np.zeros((r.shape[0])).astype('bool')
    sel[labels] = True
    data['r'] = r[labels,:]
    data['sel'][data['sel']] = (sel)

     # adicionando info ao log
    linha = '\n selaçao de ' + str(k) + ' pixels de maneira aleatória'
    print(linha,end='')
    data['log'] = np.char.add(data['log'],linha)
    return data


def dsample(data):
    import numpy as np
    n = 2
    sel = np.ones((data['dx'],data['dy']))
    XX = list(range(0,sel.shape[0]-1,n));
    YY = list(range(0,sel.shape[1]-2,n));
    sel[XX,:] = 0
    sel[:,YY] = 0
    sel = sel.reshape(-1,)
    sel = sel.astype('bool')
    data['r'] = data['r'][sel,:] 
    data['sel'][data['sel']] = (sel) 
    return data
