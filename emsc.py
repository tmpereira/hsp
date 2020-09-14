' submodulo para correçoes emsc'
def base(data,polyorder,norm = False):
    import numpy as np
    data['emsc_base_polyorder'] = polyorder
    r = data['r'].copy()
    meanspc = r.mean(0).reshape(-1,1)
    base = np.linspace(-1,1,r.shape[1]).reshape(-1,1)
    polyorder = np.arange(0,polyorder+1).reshape(1,-1)
    base =np.tile(base,(1,polyorder.shape[1]))
    base = base ** polyorder
    XX = np.hstack((meanspc,base))
    beta = np.linalg.lstsq(XX,r.T,rcond=-1)[0]
    spccorr = r - (XX[:,1:].dot(beta[1:,:]).T)
    if norm:
        div = np.tile(beta[0,:].T.reshape(-1,1),(1,spccorr.shape[1]))
        spccorr = spccorr/div
    data['r'] = spccorr
    data['EMSC_model'] = XX
    data['EMSC_coeff'] = beta.T
    data['EMSC_polyorder'] = polyorder
    return data

def h2o(data,h2o,npcs,polyorder):
  from sklearn.decomposition import PCA
  import numpy as np
  h2or = h2o['r']
  datar = data['r']
  #npcs = 1
  #polyorder = 0
  meanh2or = h2or.mean(0)
  meandatar = datar.mean(0)

  sel = data['wn'] < 1300
  # print(sel.shape)
  pca = PCA(n_components=npcs)
  pca.fit(h2or-meanh2or)
  coeff = pca.components_.T

  meanspc = datar.mean(0)
  base = np.linspace(-1,1,datar.shape[1]).reshape(-1,1)
  polyorder = np.arange(0,polyorder+1).reshape(1,-1)
  base =np.tile(base,(1,polyorder.shape[1]))
  base = base ** polyorder
  # print(base.shape)
  meanh2or[sel] = 0 
  coeff[sel,:] = 0
  XX = np.column_stack((meanspc,base,meanh2or,coeff))

  beta = np.linalg.lstsq(XX,datar.T,rcond=-1)[0]

  spccorr = datar - (XX[:,1:].dot(beta[1:,:]).T)
  div = np.tile(beta[0,:].T.reshape(-1,1),(1,spccorr.shape[1]))
  spccorr = spccorr/div
  data['EMSC_model'] = XX
  data['EMSC_coeff'] = beta.T
  data['EMSC_npcs'] = npcs
  data['EMSC_polyorder'] = polyorder
  data['r'] = spccorr
  return data


def parafin(data,para,npcs,polyorder):
  from sklearn.decomposition import PCA
  import numpy as np
  h2or = para['r']
  datar = data['r']
  meanh2or = h2or.mean(0)
  meandatar = datar.mean(0)

  sel = (data['wn'] < 1300) | (data['wn'] > 1500)
  # print(sel.shape)
  pca = PCA(n_components=npcs)
  pca.fit(h2or-meanh2or)
  coeff = pca.components_.T

  meanspc = datar.mean(0)
  base = np.linspace(-1,1,datar.shape[1]).reshape(-1,1)
  polyorder = np.arange(0,polyorder+1).reshape(1,-1)
  base =np.tile(base,(1,polyorder.shape[1]))
  base = base ** polyorder
  # print(base.shape)
  meanh2or[sel] = 0 
  coeff[sel,:] = 0
  XX = np.column_stack((meanspc,base,meanh2or,coeff))

  beta = np.linalg.lstsq(XX,datar.T,rcond=-1)[0]

  spccorr = datar - (XX[:,1:].dot(beta[1:,:]).T)
  div = np.tile(beta[0,:].T.reshape(-1,1),(1,spccorr.shape[1]))
  spccorr = spccorr/div
  data['EMSC_model'] = XX
  data['EMSC_coeff'] = beta.T
  data['EMSC_npcs'] = npcs
  data['EMSC_polyorder'] = polyorder
  data['r'] = spccorr
  return data