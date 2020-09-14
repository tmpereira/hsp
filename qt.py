# submodulo para  qualit test do hsp
def area(data,ini,fim,a,b):
    import numpy as np
    oldnspc = data['r'].shape[0]
    sel = np.logical_and(data['wn'] > int(ini),data['wn'] < int(fim))
    r = data['r'][:,sel]
    area = np.trapz(r)  
    sel = np.logical_and(area > float(a),area < float(b))
    data['r'] = data['r'][sel,:]   
    data['sel'][data['sel']] = (sel) 
    newnspc = data['r'].shape[0]
      # adicionando info ao log
    linha = '\n teste de qualidade usando a area'
    linha = linha + '\n região '+ str(ini) + ' até ' + str(fim)
    linha = linha + '\n min_value: '+ str(a) +'\n max_value: ' + str(b)
    linha = linha + '\n espectros removidos: '+ str(oldnspc-newnspc)
    
    print(linha,end='')
    data['log'] = np.char.add(data['log'],linha)
    return data

def intt(data,ini,a,b):
    import numpy as np
    import matplotlib.pyplot as plt
    oldnspc = data['r'].shape[0]
    sel = data ['wn'] == ini
    area = data['r'][:,sel]
    sel = np.logical_and(area > float(a),area < float(b))
    sel = np.reshape(sel,(-1,))
    data['r'] = data['r'][sel,:] 
    data['sel'][data['sel']] = (sel)
    newnspc = data['r'].shape[0]
    # adicionando info ao log
    linha = '\n teste de qualidade usando a intensidade'
    linha = linha + '\n pico em '+ str(ini) 
    linha = linha + '\n min_value: '+ str(a) +'\n max_value: ' + str(b)
    linha = linha + '\n espectros removidos: '+ str(oldnspc-newnspc)
    print(linha,end='')
    data['log'] = np.char.add(data['log'],linha)
    return data

def emsc(data,ini,a,b):
    import numpy as np
    import matplotlib.pyplot as plt
    oldnspc = data['r'].shape[0]
    area = data['EMSC_coeff'][:,ini];
    sel = np.logical_and(area > float(a),area < float(b))
    sel = np.reshape(sel,(-1,))
    data['r'] = data['r'][sel,:] 
    data['sel'][data['sel']] = (sel) 
    newnspc = data['r'].shape[0]
    # adicionando info ao log
    linha = '\n teste de qualidade usando o coeficientes do EMSC'
    linha = linha + '\n coef Número '+ str(ini) 
    linha = linha + '\n min_value: '+ str(a) +'\n max_value: ' + str(b)
    linha = linha + '\n espectros removidos: '+ str(oldnspc-newnspc)
    print(linha,end='')
    return data

def mean(data,ini1,fim1,a,b):
    import numpy as np
    oldnspc = data['r'].shape[0]
    sel = np.logical_and(data['wn'] > ini1,data['wn'] < fim1)
    r1 = data['r'][:,sel]
    media = np.mean(r1,axis=0).reshape(-1,1)
    meansvalue = np.zeros((data['r'].shape[0]))
    y = r1[:,:].T
    xx= np.vstack((r1.mean(axis=0),np.ones_like(r1[1,:]))).T
    alpha= np.linalg.lstsq(xx, y,rcond =-1)[0][0].T
    area = (alpha)
    sel = np.logical_and(area > float(a),area < float(b))
    data['r'] = data['r'][sel,:]   
    data['sel'][data['sel']] = (sel) 
    newnspc = data['r'].shape[0]
      
    # adicionando info ao log
    linha = '\n teste de qualidade usando a meanspc'
    linha = linha + '\n região '+ str(ini1) + ' até ' + str(fim1)
    linha = linha + '\n min_value: '+ str(a) +'\n max_value: ' + str(b)
    linha = linha + '\n espectros removidos: '+ str(oldnspc-newnspc)
    
    print(linha,end='')
    data['log'] = np.char.add(data['log'],linha)
    return data


