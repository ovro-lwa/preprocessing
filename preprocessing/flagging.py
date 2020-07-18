import os
import numpy as np
from scipy.ndimage import filters
import casatools


def flag_bad_chans(msfile: str, band: str = None, usedatacol: bool = False, applyflags: bool = True, writeflagfile: bool = True):
    """
    Input: msfile
    Finds remaining bad channels and flags those in the measurement set.
    band is inferred from name if not given explicitly as int.
    applyflags and writeflagfile control whether to apply or write flags as text file.
    Optionally writes out text file that lists flags that were applied.
    """

    outfile = os.path.dirname(os.path.abspath(msfile)) + '/flags.chans'

    tb = casatools.table()
    tb.open(msfile, nomodify=False)
    tcross  = tb.query('ANTENNA1!=ANTENNA2')

    if usedatacol:
        datacol = tcross.getcol('DATA')
    else:
        try:
            datacol = tcross.getcol('CORRECTED_DATA')
        except RuntimeError:
            print('No CORRECTED_DATA found. Trying DATA column')
            datacol = tcross.getcol('DATA')

    # merge flags
    ms = casatools.ms()
    ms.open(msfile)
    nspw = len(ms.getspectralwindowinfo())
    ms.close()

    flagcol = tcross.getcol('FLAG')
    npol, nchan, nblnspw = flagcol.shape
    nbl = nblnspw//nspw
    print('Data shape: {0} bls, {1} chans/spw, {2} spw, {3} pol'.format(nbl, nchan, nspw, npol))

    datacol = datacol.reshape(npol, nchan*nspw, nbl)  # unwrap spw
    flagcol = flagcol.reshape(npol, nchan*nspw, nbl)  # unwrap spw
    flagarr = np.rollaxis(flagcol[0,:,:] | flagcol[1,:,:] | flagcol[2,:,:] | flagcol[3,:,:], 1)
    print('{0}% of data already flagged'.format(100*np.count_nonzero(flagarr)/flagarr.size))

    datacolxx = np.rollaxis(datacol[0], 1)
    datacolxy = np.rollaxis(datacol[1], 1)
    datacolyx = np.rollaxis(datacol[2], 1)
    datacolyy = np.rollaxis(datacol[3], 1)

    datacolxxamp = np.ma.masked_invalid(np.abs(datacolxx)**2)
    datacolyyamp = np.ma.masked_invalid(np.abs(datacolyy)**2)
    datacolxyamp = np.ma.masked_invalid(np.abs(datacolxy)**2)
    datacolyxamp = np.ma.masked_invalid(np.abs(datacolyx)**2)

    mask = np.ma.make_mask(flagarr)
    mask = mask | datacolxxamp.mask | datacolyyamp.mask | datacolxyamp.mask | datacolyxamp.mask

    datacolxxamp_mask = np.ma.masked_array(datacolxxamp, mask=mask, fill_value=np.nan)
    datacolyyamp_mask = np.ma.masked_array(datacolyyamp, mask=mask, fill_value=np.nan)
    datacolxyamp_mask = np.ma.masked_array(datacolxyamp, mask=mask, fill_value=np.nan)
    datacolyxamp_mask = np.ma.masked_array(datacolyxamp, mask=mask, fill_value=np.nan)

    maxxx = np.ma.max(datacolxxamp_mask,axis=0)
    maxyy = np.ma.max(datacolyyamp_mask,axis=0)
    maxxy = np.ma.max(datacolxyamp_mask,axis=0)
    maxyx = np.ma.max(datacolyxamp_mask,axis=0)
    meanxx = np.ma.mean(datacolxxamp_mask,axis=0)
    meanyy = np.ma.mean(datacolyyamp_mask,axis=0)
    meanxy = np.ma.mean(datacolxyamp_mask,axis=0)
    meanyx = np.ma.mean(datacolyxamp_mask,axis=0)

    maxxx_medfilt = filters.median_filter(maxxx,size=10)
    maxyy_medfilt = filters.median_filter(maxyy,size=10)
    maxxy_medfilt = filters.median_filter(maxxy,size=10)
    maxyx_medfilt = filters.median_filter(maxyx,size=10)

    maxxx_norm = maxxx/maxxx_medfilt
    maxyy_norm = maxyy/maxyy_medfilt
    maxxy_norm = maxxy/maxxy_medfilt
    maxyx_norm = maxyx/maxyx_medfilt
    
    maxxx_norm_stdfilt = filters.generic_filter(maxxx_norm, np.ma.std, size=25)
    maxyy_norm_stdfilt = filters.generic_filter(maxyy_norm, np.ma.std, size=25)
    maxxy_norm_stdfilt = filters.generic_filter(maxxy_norm, np.ma.std, size=25)
    maxyx_norm_stdfilt = filters.generic_filter(maxyx_norm, np.ma.std, size=25)
    maxvalxx  = 1 - 10*np.ma.min(maxxx_norm_stdfilt)
    maxval2xx = 1 + 10*np.ma.min(maxxx_norm_stdfilt)
    maxvalyy  = 1 - 10*np.ma.min(maxyy_norm_stdfilt)
    maxval2yy = 1 + 10*np.ma.min(maxyy_norm_stdfilt)
    maxvalxy  = 1 - 6*np.ma.min(maxxy_norm_stdfilt)
    maxval2xy = 1 + 6*np.ma.min(maxxy_norm_stdfilt)
    maxvalyx  = 1 - 6*np.ma.min(maxyx_norm_stdfilt)
    maxval2yx = 1 + 6*np.ma.min(maxyx_norm_stdfilt)
    meanxx_stdfilt = filters.generic_filter(meanxx, np.ma.std, size=25)
    meanyy_stdfilt = filters.generic_filter(meanyy, np.ma.std, size=25)
    meanxy_stdfilt = filters.generic_filter(meanxy, np.ma.std, size=25)
    meanyx_stdfilt = filters.generic_filter(meanyx, np.ma.std, size=25)

    # bad channels tend to have maxness values close to zero or slightly negative, compared to
    # good channels, which have significantly positive maxs, or right-maxed distributions.
    #flaglist = np.where( (maxxx < 1) | (maxyy < 1)  )
    flaglist = np.where( (maxxx_norm < maxvalxx) | (maxyy_norm < maxvalyy) |   \
                         (maxxx_norm > maxval2xx) | (maxyy_norm > maxval2yy) | \
                         (meanxx > np.ma.median(meanxx)+100*np.ma.min(meanxx_stdfilt))  | \
                         (meanyy > np.ma.median(meanyy)+100*np.ma.min(meanyy_stdfilt)) | \
                         (maxxy_norm < maxvalxy) | (maxyx_norm < maxvalyx) | \
                         (maxxy_norm > maxval2xy) | (maxyx_norm > maxval2yx) | \
                         (meanxy > np.ma.median(meanxy)+100*np.ma.min(meanxy_stdfilt)) | \
                         (meanyx > np.ma.median(meanyx)+100*np.ma.min(meanyx_stdfilt)) ) 

    print('Flagged {0} of {1} channels'.format(len(flaglist[0]), nchan*nspw))

    ################################################

    if flaglist[0].size > 0 and writeflagfile:
        # turn flaglist into text file of channel flags
        chans    = np.arange(0, nchan*nspw)
        chanlist = chans[flaglist]

        if band is None:
            try:
                band = int(msfile.split('_')[0])
            except ValueError:
                print('Cannot infer band from filename. Assuming 0.')
                band = 0

        with open(outfile, 'w') as f:
            for chan in chanlist:
                f.write('%02d:%03d\n' % (band, chan))

    # write flags into FLAG column
    if applyflags:
        flagcol_altered = tcross.getcol('FLAG').reshape(npol, nchan*nspw, nbl)
        flagcol_altered[:, flaglist, :] = 1
        tcross.putcol('FLAG', flagcol_altered.reshape(npol, nchan, nbl*nspw))

    flagcol = tb.getcol('FLAG')
    print('{0}% of data now flagged'.format(100*np.count_nonzero(flagcol)/flagcol.size))
    tb.close()

    if os.path.exists(outfile):
        if writeflagfile:
            print("Wrote flag file {0}".format(outfile))
        return outfile
    else:
        if writeflagfile:
            print("No flag file written")
        return None


def flag_bad_ants(msfile: str, threshold: float = 0.02, applyflags: bool = True, writeflagfile: bool = True):
    """
    Input: msfile
    Optionally can define threshold for excess variance in antenna noise.
    applyflags and writeflagfile control whether to apply or write flags as text file.
    If flag file written, function will return path to text file that lists flags.
    """

    outfile = os.path.dirname(os.path.abspath(msfile)) + '/flags.ants'

    tb = casatools.table()
    tb.open(msfile, nomodify=False)
    flagcol = tb.getcol('FLAG')
    flagarr = np.rollaxis(flagcol[0,:,:] | flagcol[1,:,:] | flagcol[2,:,:] | flagcol[3,:,:], 1)
    print('{0}% of data already flagged'.format(100*np.count_nonzero(flagarr)/flagarr.size))
    ms = casatools.ms()
    ms.open(msfile)
    nspw = len(ms.getspectralwindowinfo())
    ms.close()

    tautos = tb.query('ANTENNA1=ANTENNA2')
    tband = tautos.getcol('DATA')
    npol, nchan, nantnspw = tband.shape
    nant = nantnspw//nspw
    nbl = nant*(nant-1)//2
    tband = tband.reshape(npol, nchan*nspw, nant)  # unwrap spw

    flagcol = tautos.getcol('FLAG')
    flagcol = flagcol.reshape(npol, nchan*nspw, nant)  # unwrap spw
    flagarr = np.rollaxis(flagcol[0,:,:] | flagcol[1,:,:] | flagcol[2,:,:] | flagcol[3,:,:], 1)

    print('Data shape: {0} ants, {1} chans/spw, {2} spw, {3} pol'.format(nant, nchan, nspw, npol))
    datacolxx = np.rollaxis(tband[0], 1)
    datacolyy = np.rollaxis(tband[3], 1)

    mask = np.ma.make_mask(flagarr)
    datacolxxamp = np.ma.masked_array(np.abs(datacolxx)**2, mask=mask)
    datacolyyamp = np.ma.masked_array(np.abs(datacolyy)**2, mask=mask)

    datacolxxampdb = 10*np.ma.log10(datacolxxamp/1.e2)
    datacolyyampdb = 10*np.ma.log10(datacolyyamp/1.e2)

    # median value for every antenna
    medamp_perantx = np.ma.median(datacolxxampdb, axis=1)
    medamp_peranty = np.ma.median(datacolyyampdb, axis=1)

    print('Median amp per ant x/y:')
    print('med(med(Amp_x))={0}, std(med(Amp_x))={1}, med(med(Amp_y))={2}, std(med(Amp_y))={3}'.format(np.ma.median(medamp_perantx), np.ma.std(medamp_perantx), np.ma.median(medamp_peranty), np.ma.std(medamp_peranty)))

    # get flags based on deviation from median amp
    xthresh_pos = np.ma.median(medamp_perantx) + np.ma.std(medamp_perantx)
    xthresh_neg = np.ma.median(medamp_perantx) - 2*np.ma.std(medamp_perantx)
    ythresh_pos = np.ma.median(medamp_peranty) + np.ma.std(medamp_peranty)
    ythresh_neg = np.ma.median(medamp_peranty) - 2*np.ma.std(medamp_peranty)
    flags = np.where( (medamp_perantx > xthresh_pos) | (medamp_perantx < xthresh_neg) |\
                      (medamp_peranty > ythresh_pos) | (medamp_peranty < ythresh_neg) )
    print('Ant flags ({0} in first pass): {1}.'.format(len(flags[0]), flags[0]))

    # use unflagged antennas to generate median spectrum
    datacolxxampdb.mask[flags[0]] = True
    datacolyyampdb.mask[flags[0]] = True

    medamp_specx = np.ma.median(datacolxxampdb, axis=0)
    medamp_specy = np.ma.median(datacolyyampdb, axis=0)

    stdarrayx = np.ma.std(datacolxxampdb/medamp_specx, axis=1)
    stdarrayy = np.ma.std(datacolyyampdb/medamp_specy, axis=1)
#    stdarrayy = np.array( [np.ma.std(spec/medamp_specy) for spec in datacolyyampdb] )
    print(stdarrayx, stdarrayy)

    flags2 = np.where( (stdarrayx > threshold) | (stdarrayy > threshold) )
    flagsall = np.sort(np.append(flags, flags2))
    flagsallstr = ",".join([str(flag) for flag in flagsall])
    print('Ant flags ({0} in second pass): {1}.'.format(len(flagsallstr.split(',')), flagsallstr))

    if writeflagfile:
        with open(outfile, 'w') as f:
            f.write(flagsallstr)
    
    if applyflags:
        ms = casatools.ms()
        ms.open(msfile)
        ant1 = ms.getdata('antenna1')['antenna1']
        ant2 = ms.getdata('antenna2')['antenna2']

        blflags = []
        for badant in flagsall:
            blflags += np.where((badant == ant1) | (badant == ant2))[0].tolist()

        print('blflags:', blflags)
        flagcol_altered = tb.getcol('FLAG')
        print(flagcol_altered.shape)
        flagcol_altered[:, :, blflags] = 1
        tb.putcol('FLAG', flagcol_altered)

    flagcol = tb.getcol('FLAG')
    print('{0}% of data now flagged'.format(100*np.count_nonzero(flagcol)/flagcol.size))
    tb.close()

    if os.path.exists(outfile):
        if writeflagfile:
            print("Wrote flag file {0}".format(outfile))
        return outfile
    else:
        if writeflagfile:
            print("No flag file written")
        return None


def unflag(msfile: str):
    """ Remove all flags from msfile
    """ 

    import casatasks
    casatasks.flagdata(msfile, mode='unflag')

    tb = casatools.table()
    tb.open(msfile, nomodify=False)
    flagcol = tb.getcol('FLAG')
    print('{0}% of data flagged'.format(100*np.count_nonzero(flagcol)/flagcol.size))
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
    tb.close()
