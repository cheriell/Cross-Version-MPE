


# Schubert dataset split definitions
train_songs_schubert = ['D911-01', 'D911-02', 'D911-03', 'D911-04', 'D911-05', 'D911-06', 'D911-07', 'D911-08', 'D911-09', 'D911-10', 'D911-11', 'D911-12', 'D911-13', ]
val_songs_schubert = ['D911-14', 'D911-15', 'D911-16', ]
test_songs_schubert = ['D911-17', 'D911-18', 'D911-19', 'D911-20', 'D911-21', 'D911-22', 'D911-23', 'D911-24']
train_versions_schubert = ['AL98', 'FI55', 'FI80', 'OL06', 'QU98']
val_versions_schubert = ['FI66', 'TR99']
test_versions_schubert = ['HU33', 'SC06']


# Wagner dataset split definitions
train_songs_wagner  = ['WWV086C-1', 'WWV086C-2', 'WWV086C-3', 'WWV086D-0', 'WWV086D-1', 'WWV086D-2', 'WWV086D-3']
val_songs_wagner = ['WWV086A']
test_songs_wagner = ['WWV086B-1']  # the song with accurate pitch annotations. 'WWV086B-2', 'WWV086B-3' are not used since there can be similar motifs
train_versions_wagner = ['MEMBRAN2013', 'DG2012', 'PHILIPS2006', 'EMI2012', 'DECCA2012', 'DG2013', 'DECCA2008', 'OEHMS2013', 'NAXOS2003', 'PROFIL2013', 'SONY2012', 'MEMBRAN1995']
val_versions_wagner = ['ZYX2012', 'EMI2011', 'ORFEO2010']   # the original test versions in the old splits, these are the public ones.
test_versions_wagner = ['WC2009', 'EMI2008', 'DG1998']  # the versions with accurate pitch annotations, open these features?


# split for n_versions experiments （Wagner dataset)
train_n_versions = ['MEMBRAN2013', 'DG2012', 'PHILIPS2006','EMI2012', 'DECCA2012', 'DG2013', 'DECCA2008', 'OEHMS2013']
val_n_versions = ['ZYX2012', 'EMI2011', 'ORFEO2010', 'NAXOS2003', 'PROFIL2013', 'SONY2012', 'MEMBRAN1995']

