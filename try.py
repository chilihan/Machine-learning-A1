
# print("------------------------------")
#
# print("X:",X,X.shape)
# print ("this is x:", X[1],X[1].shape)
d = X.shape[1] #shape =36
samplesize = X.shape[0] #shape = 6500
#samplesize = 20
hidu = len(W_out) #shape = 100
dE_dWout = np.zeros(hidu) #shape = (100,)
dE_dWhid = np.zeros((d,hidu)) #shape  = (36,100)

dE_dbout = 0 #()
dE_dbhid = np.zeros((100,))
# dE_dbhid = np.zeros()

#
#test dE_dWout
x = X[0] #(36,)
print ("x:",x.shape)
ytrue = y[0] #()
print ("ytrue:",ytrue.shape)
y_hid = p_y_given_x(W_hid, b_hid, x) #(100,)
y_out = p_y_given_x(W_out, b_out, y_hid) #shape = ()

print ("yhid:",y_hid,y_hid.shape)



dE_dyout = np.divide(ytrue, y_out) #()
print ("dedout:", dE_dyout, dE_dyout.shape)

dyout_dzout = np.multiply(np.subtract(1,y_out), y_out) # ()
dE_dzout = np.multiply(dE_dyout,dyout_dzout) #()
print ("dyout_dzout:", dyout_dzout,dyout_dzout.shape)
print ("dE_dzout:", dE_dzout,dE_dzout.shape)

dE_dWoutchange = np.multiply(dE_dzout,y_hid) #(100,)
dE_dWout = np.add(dE_dWout, dE_dWoutchange) #(100,)+(100,)=(100,)
print ("dE_dWoutchange:", dE_dWoutchange, dE_dWoutchange.shape)
print ("dE_dWout:", dE_dWout,dE_dWout.shape)

#test dE_dWhid
dzout_dyhid = W_out #(100,)
print ("dzout_dyhid:", dzout_dyhid,dzout_dyhid.shape)

yhid_1 = 1-y_hid #(100,)
print ("1_yhid", yhid_1,yhid_1.shape)
dyhid_dzhid = y_hid*yhid_1  #(100,)
print ("dyhid_dzhid:",dyhid_dzhid,dyhid_dzhid.shape)

#dzo/dyh * dyh/dzh = dzo/dzh
dzout_dzhid = dzout_dyhid * dyhid_dzhid
dE_dzhid = dE_dzout * np.array(dzout_dzhid)
print ("dzout_dzhid:",dzout_dzhid,dzout_dzhid.shape)
print ("dE_dzhid:",dE_dzhid,dE_dzhid.shape)

dE_dzhid_mat = np.reshape(dE_dzhid,(1,len(dE_dzhid)))
print ("dE_dzhid_vec:",dE_dzhid_mat,dE_dzhid_mat.shape)
x_vec = np.reshape(x,(len(x),1))
print("x_vec:",x_vec,x_vec.shape)
dE_dWhidchange = np.dot(x_vec,dE_dzhid_mat) #(36, 100)
print("dE_dWhidchange:",dE_dWhidchange,dE_dWhidchange.shape)
dE_dWhid = np.add(dE_dWhid,dE_dWhidchange)
print ("dE_dWhid:",dE_dWhid,dE_dWhid.shape)
dE_dWhid = (1.0/(samplesize))*np.array(dE_dWhid)
dE_dWout = (1.0/(samplesize))*np.array(dE_dWout)



print ("dyh_dzh",dyhdzh, dyhdzh.shape)
