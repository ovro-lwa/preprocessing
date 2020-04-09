import os
import numpy as np
from scipy.ndimage import filters
import casatools

def flag_bad_chans(msfile: str, band: str, usedatacol: bool = False):
    """
    Input: msfile
    Finds remaining bad channels and flags those in the measurement set. Also writes out text file that lists
    flags that were applied.
    """

    tb = casatools.table()
    tb.open(msfile, nomodify=False)

    tcross  = tb.query('ANTENNA1!=ANTENNA2')
    if usedatacol:
        datacol = tcross.getcol('DATA')
    else:
        datacol = tcross.getcol('CORRECTED_DATA')
    flagcol = tcross.getcol('FLAG')
    
    datacolxx = datacol[:,:,0]
    datacolyy = datacol[:,:,3]
    datacolxy = datacol[:,:,1]
    datacolyx = datacol[:,:,2]

    datacolxxamp = np.sqrt( np.real(datacolxx)**2. + np.imag(datacolxx)**2. )
    datacolyyamp = np.sqrt( np.real(datacolyy)**2. + np.imag(datacolyy)**2. )
    datacolxyamp = np.sqrt( np.real(datacolxy)**2. + np.imag(datacolxy)**2. )
    datacolyxamp = np.sqrt( np.real(datacolyx)**2. + np.imag(datacolyx)**2. )

    #flagarr = flagcol[:,:,0] | flagcol[:,:,3]   # probably unnecessary since flags are never pol-specific,
                                                 # but doing this just in cases
    flagarr = flagcol[:,:,0] | flagcol[:,:,1] | flagcol[:,:,2] | flagcol[:,:,3]

    datacolxxamp_mask = np.ma.masked_array(datacolxxamp, mask=flagarr, fill_value=np.nan)
    datacolyyamp_mask = np.ma.masked_array(datacolyyamp, mask=flagarr, fill_value=np.nan)
    datacolxyamp_mask = np.ma.masked_array(datacolxyamp, mask=flagarr, fill_value=np.nan)
    datacolyxamp_mask = np.ma.masked_array(datacolyxamp, mask=flagarr, fill_value=np.nan)

    maxxx = np.max(datacolxxamp_mask,axis=0)
    maxyy = np.max(datacolyyamp_mask,axis=0)
    maxxy = np.max(datacolxyamp_mask,axis=0)
    maxyx = np.max(datacolyxamp_mask,axis=0)
    meanxx = np.mean(datacolxxamp_mask,axis=0)
    meanyy = np.mean(datacolyyamp_mask,axis=0)
    meanxy = np.mean(datacolxyamp_mask,axis=0)
    meanyx = np.mean(datacolyxamp_mask,axis=0)

    maxxx_medfilt = filters.median_filter(maxxx,size=10)
    maxyy_medfilt = filters.median_filter(maxyy,size=10)
    maxxy_medfilt = filters.median_filter(maxxy,size=10)
    maxyx_medfilt = filters.median_filter(maxyx,size=10)

    maxxx_norm = maxxx/maxxx_medfilt
    maxyy_norm = maxyy/maxyy_medfilt
    maxxy_norm = maxxy/maxxy_medfilt
    maxyx_norm = maxyx/maxyx_medfilt
    
    maxxx_norm_stdfilt = filters.generic_filter(maxxx_norm, np.std, size=25)
    maxyy_norm_stdfilt = filters.generic_filter(maxyy_norm, np.std, size=25)
    maxxy_norm_stdfilt = filters.generic_filter(maxxy_norm, np.std, size=25)
    maxyx_norm_stdfilt = filters.generic_filter(maxyx_norm, np.std, size=25)
    maxvalxx  = 1 - 10*np.min(maxxx_norm_stdfilt)
    maxval2xx = 1 + 10*np.min(maxxx_norm_stdfilt)
    maxvalyy  = 1 - 10*np.min(maxyy_norm_stdfilt)
    maxval2yy = 1 + 10*np.min(maxyy_norm_stdfilt)
    maxvalxy  = 1 - 6*np.min(maxxy_norm_stdfilt)
    maxval2xy = 1 + 6*np.min(maxxy_norm_stdfilt)
    maxvalyx  = 1 - 6*np.min(maxyx_norm_stdfilt)
    maxval2yx = 1 + 6*np.min(maxyx_norm_stdfilt)
    meanxx_stdfilt = filters.generic_filter(meanxx, np.std, size=25)
    meanyy_stdfilt = filters.generic_filter(meanyy, np.std, size=25)
    meanxy_stdfilt = filters.generic_filter(meanxy, np.std, size=25)
    meanyx_stdfilt = filters.generic_filter(meanyx, np.std, size=25)

    # bad channels tend to have maxness values close to zero or slightly negative, compared to
    # good channels, which have significantly positive maxs, or right-maxed distributions.
    #flaglist = np.where( (maxxx < 1) | (maxyy < 1)  )
    flaglist = np.where( (maxxx_norm < maxvalxx) | (maxyy_norm < maxvalyy) |   \
                         (maxxx_norm > maxval2xx) | (maxyy_norm > maxval2yy) | \
                         (meanxx > np.median(meanxx)+100*np.min(meanxx_stdfilt))  | \
                         (meanyy > np.median(meanyy)+100*np.min(meanyy_stdfilt)) | \
                         (maxxy_norm < maxvalxy) | (maxyx_norm < maxvalyx) | \
                         (maxxy_norm > maxval2xy) | (maxyx_norm > maxval2yx) | \
                         (meanxy > np.median(meanxy)+100*np.min(meanxy_stdfilt)) | \
                         (meanyx > np.median(meanyx)+100*np.min(meanyx_stdfilt)) ) 

    ################################################

    if flaglist[0].size > 0:
        # turn flaglist into text file of channel flags
        textfile = os.path.splitext(os.path.abspath(msfile))[0]+'.chans'
        chans    = np.arange(0,109)
        chanlist = chans[flaglist]
        with open(textfile, 'w') as f:
            for chan in chanlist:
                f.write('%02d:%03d\n' % (np.int(band),chan))

        # write flags into FLAG column
        flagcol_altered = tb.getcol('FLAG')
        flagcol_altered[:,flaglist,:] = 1
        tb.putcol('FLAG', flagcol_altered)
        #os.system('apply_sb_flags_single_band_ms2.py %s %s %02d' % (textfile,msfile,np.int(band)) )
    tb.close()


def flag_bad_ants(msfile: str):
    """
    Input: msfile
    Returns list of antennas to be flagged based on autocorrelations.
    """

    tb = casatools.table()
    tb.open(msfile, nomodify=False)

    tautos  = tb.query('ANTENNA1=ANTENNA2')
    
    # iterate over antenna, 1-->256
    nant = 256
    nchan = 109
#    datacolxx = np.zeros((nant, nchan))
#    datacolyy = np.copy(datacolxx)
#    for antind in range(nant):
#        print(antind)
    tband = tautos.getcol('DATA')
#    datacolxx[antind,bandind*109:(bandind+1)*109] = tband["DATA"][:,0]
#    datacolyy[antind,bandind*109:(bandind+1)*109] = tband["DATA"][:,3]
    datacolxx = np.rollaxis(tband[0], 1)
    datacolyy = np.rollaxis(tband[3], 1)

    datacolxxamp = np.sqrt( np.real(datacolxx)**2. + np.imag(datacolxx)**2. )
    datacolyyamp = np.sqrt( np.real(datacolyy)**2. + np.imag(datacolyy)**2. )

    datacolxxampdb = 10*np.log10(datacolxxamp/1.e2)
    datacolyyampdb = 10*np.log10(datacolyyamp/1.e2)

    # median value for every antenna
    medamp_perantx = np.median(datacolxxampdb,axis=1)
    medamp_peranty = np.median(datacolyyampdb,axis=1)

    # get flags based on deviation from median amp
    xthresh_pos = np.median(medamp_perantx) + np.std(medamp_perantx)
    xthresh_neg = np.median(medamp_perantx) - 2*np.std(medamp_perantx)
    ythresh_pos = np.median(medamp_peranty) + np.std(medamp_peranty)
    ythresh_neg = np.median(medamp_peranty) - 2*np.std(medamp_peranty)
    flags = np.where( (medamp_perantx > xthresh_pos) | (medamp_perantx < xthresh_neg) |\
                      (medamp_peranty > ythresh_pos) | (medamp_peranty < ythresh_neg) )

    # use unflagged antennas to generate median spectrum
    flagmask = np.zeros((nant,nchan))
    flagmask[flags[0],:] = 1
    datacolxxampdb_mask = np.ma.masked_array(datacolxxampdb, mask=flagmask, fill_value=np.nan)
    datacolyyampdb_mask = np.ma.masked_array(datacolyyampdb, mask=flagmask, fill_value=np.nan)

    medamp_allantsx = np.median(datacolxxampdb_mask,axis=0)
    medamp_allantsy = np.median(datacolyyampdb_mask,axis=0)

    stdarrayx = np.array( [np.std(antarr/medamp_allantsx) for antarr in datacolxxampdb_mask] )
    stdarrayy = np.array( [np.std(antarr/medamp_allantsy) for antarr in datacolyyampdb_mask] )
    
    # this threshold was manually selected...should be changed to something better at some point
    flags2 = np.where( (stdarrayx > 0.02) | (stdarrayy > 0.02) )

    flagsall = np.sort(np.append(flags,flags2))
    flagsallstr = [str(flag) for flag in flagsall]
    flagsallstr2 = ",".join(flagsallstr)

    antflagfile = os.path.dirname(os.path.abspath(msfile)) + '/flag_bad_ants.ants'
    with open(antflagfile,'w') as f:
        f.write(flagsallstr2)
    
    tb.close()


def merge_flags(msfile1: str, msfile2: str):
    """ Read two flag tables, merge and write them back into both files
    """

    # open two flag tables
    tb1 = casatools.table(readonly=False)
    tb2 = casatools.table(readonly=False)
    tb1.open(msfile1)
    tb2.open(msfile2)

    # merge two flag cols
    flagcol1 = tb1.getcol('FLAG')
    flagcol2 = tb2.getcol('FLAG')
    flagcol = flagcol1 | flagcol2

    # put them back in to both
    tb1.putcol('FLAG', flagcol)
    tb2.putcol('FLAG', flagcol)


def write_to_flag_column(msfile: str, flag_npy: str):
    """ Load numpy flag array and write it to ms flag column.
    Uses or operator to flag all originally flagged plus numpy flags.
    """

    tb = casatools.table(readonly=False)
    tb.open(msfile)
    flagcol = np.load(flag_npy)
    assert flagcol.shape == tb.getcol('FLAG').shape, 'Flag file and measurement set have different shapes'
    tb.putcol('FLAG', flagcol | tb.getcol('FLAG'))
