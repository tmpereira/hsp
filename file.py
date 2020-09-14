'''submodulo que faz a leitura de arquivos dos principais micros-FTIR
criado no dia 01-09-2020
'''
# função que importa um arquivo fsm da perkin elmer
def get_fsm_files(path):
    import glob, os
    arq = []
    os.chdir(path)
    for file in glob.glob("*.fsm"):
        arq.append(path +  file)
    return arq

# funçao que importa um arquivo FSM
    
def fsm(arq):
    from specio import specread
    import numpy as np
    ver = specread(arq)
    meta = ver.meta
    data = {'r':np.fliplr(ver.amplitudes)}
    data['wn'] = np.flipud(ver.wavelength)
    data['r'] = np.log10(0.01*data['r'])
    data['r'] = -1*data['r']
    dx = data['dy'] = np.array(meta['n_x']) 
    dy = data['dx'] = np.array(meta['n_y'])
    data['filename'] = np.array(meta['filename']) 
    data['sel'] = np.ones((dx*dy), dtype=bool)
    data['log'] = np.array(' abrindo o arquivo ')
    print(data['log'] ,end='')
    print(' ', arq)
    return data


def npz_save(arq,data):
    import numpy as np
    np.savez(arq,**data)


def npz_load(arq):
    import numpy as np
    f =  np.load(arq)
    dados = {}
    for i in f.files:
        dados[i] = f[i]
    return(dados)

# funçao que abre os arquivos do agilent
 
def age(path):
    import numpy as np
    import matplotlib.pyplot as plt
    import glob
    # descobrindo o numero de tiles
    dx = []
    dy = []
    for i in glob.glob(path + '*.dmd'):
         i = i[-13:-4].split('_')
         dx.append(int(i[0]))
         dy.append(int(i[1]))
    dx = np.array(dx).max()+1
    dy = np.array(dy).max()+1 
    dados ={}
    dados['dx'] = np.array(32*dx)
    dados['dy'] = np.array(32*dy)
    dados['filename'] = np.array(path.split('\\')[-2]+'.dmd')
    
    # lendo os arquivo dmt construir o vetor wn
    
    arq = glob.glob(path + '*.dmt')[0]# path + file + '.dmt'
    data = np.fromfile(arq,dtype='i4')
    npoints = data[559]
    start = data[557]
    data = np.fromfile(arq)
    steps = data[389]
    dados['wn'] = np.linspace(start*steps,(start+npoints)*steps,npoints)
    
    
    # lendo arquivos dmd
    r = np.zeros((int(32*dx),int(32*dy),int(npoints)))   
    arq = glob.glob(path + '*.dmd')
    for i in arq:
        y=int(i[-8:-4])
        x=int(i[-13:-9])
        print(i.split('\\')[-1],' file readed')
        data = np.fromfile(i,'f4')[255:]
        data = np.reshape(data,(-1,32,32))
        data = np.transpose(data,(2,1,0))
        data = data[:,::-1,:]
        r[32*x:32*(x+1),32*y:32*(y+1),:] = data
    dados['r'] = r.reshape(1024*dx*dy,-1)
    dados['sel'] =  np.ones(dados['r'].shape[0]).astype('bool')
    dados['log'] = np.array('abrindo mosaico agilent contido na pasta ' + path )
    return dados
