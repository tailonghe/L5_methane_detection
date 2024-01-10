from backend.utils import *
from backend.ee_retrieval import *
import pandas as pd
from datetime import datetime, timedelta
import geemap
import ee
import numpy as np
from sklearn.linear_model import HuberRegressor
import os
from scipy import interpolate
from scipy.interpolate import griddata
from skimage.metrics import structural_similarity as ssim
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
import proplot as pplt
import glob
import warnings
warnings.filterwarnings("ignore")
import traceback

pplt.rc.fontsize = 10
pplt.rc.abc = False
import warnings
warnings.filterwarnings("ignore")


from PlumeNet import *
import numpy as np
from radtran import setup
import radtran.radtran as rt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
pnet_model = PlumeNet(6, 1, n_classes=1).to(device)
pnet_models_weights = glob.glob('model_weights/*.h5')


def dd2dms(deg):
    d = int(deg)
    md = abs(deg - d) * 60
    m = int(md)
    sd = (md - m) * 60
    sd = int(sd)
    return "%d°%d\'%d\""%(d, m, sd)


def get_plume(lon, lat, startDate, endDate, dX=1.5, dY=1.5, do_retrieval=False, satellite='L8'):
    """
    dX/dY: distance (km) in NS/WE direction
    lon: longitude (~180 -- 180)
    lat: latitude
    startDate/endDate: string ('YYYY-MM-DD') for initial/final date
    do_retrieval: flag for calculating XCH4 using the MBSP approach
    satellite: Satellite name (L4, L5, L7, L8, and S2) 
    """
    
    # Initialize Earth Engine
    ee.Initialize()

    # Coordinate mapping for rectangle of plume
    grid_pt = (lat, lon)
    dlat = dY/110.574
    dlon = dX/111.320/np.cos(lat*0.0174532925)
    #print('dlat, dlon: ', dlat, dlon)
    W=grid_pt[1]-dlon 
    E=grid_pt[1]+dlon
    N=grid_pt[0]+dlat
    S=grid_pt[0]-dlat
    re   = ee.Geometry.Point(lon, lat)
    region = ee.Geometry.Polygon(
        [[W, N],\
        [W, S],\
        [E, S],\
        [E, N]])

    era5_region = ee.Geometry.Polygon(
            [[lon-0.01, lat+0.01],\
              [lon-0.01, lat-0.01],\
              [lon+0.01, lat-0.01],\
              [lon+0.01, lat+0.01]])

    
    print('check2')
    redband = satellite_database[satellite]['Red']
    greenband = satellite_database[satellite]['Green']
    blueband = satellite_database[satellite]['Blue']
    nirband = satellite_database[satellite]['NIR']
    swir1band = satellite_database[satellite]['SWIR1']
    swir2band = satellite_database[satellite]['SWIR2']
    cloudband = satellite_database[satellite]['Cloud']
    foldername = satellite_database[satellite]['Folder']

    # Pull the desired collection; filter date, region and bands
    # filterMetaData allows us to pick the desired Grid Reference System. Since the images appeared identicle, I picked 31SGR...
    # If the other MGRS is better, we can remove filterMetadata, reprint, and pick the other. 
    if satellite == 'Sentinel-2':
        _default_value = None
        scaleFac = 0.0001
        img_collection = get_s2_cld_col(region, startDate, endDate)  # Oct10 tlh: change to tiles covering the central point only
        if img_collection is not None:
            img_collection = img_collection.map(add_cloud_bands).select([redband,
                                            greenband,
                                            blueband,
                                            nirband,
                                            swir1band,
                                            swir2band,
                                            'cloud_prob'])
    else:
        _default_value = -999.0
        scaleFac = 1

        # Oct10 tlh: change to tiles covering the central point only
        img_collection = ee.ImageCollection('LANDSAT/%s'%foldername).filterDate(startDate, endDate).filterBounds(region).select([redband,
                                                                        greenband,
                                                                        blueband,
                                                                        nirband,
                                                                        swir1band,
                                                                        swir2band,
                                                                        cloudband])

    # initialize arrays
    chanlarr = None
    zarr = None
    lonarr = None
    latarr = None
    date_list2 = []
    u10m, v10m = [], []
    SZA = []
    
    if img_collection is None:
        id_list, date_list = [], []
        pass

    # convert to list of images
    collectionList = img_collection.toList(img_collection.size())



    if img_collection.size().getInfo() == 0:
        id_list, date_list = [], []
        pass
    else:
        ### DATELIST for plumes ###
        methaneAlt = img_collection.getRegion(re,50).getInfo()
        id_list    = pd.DataFrame(methaneAlt)
        headers    = id_list.iloc[0]
        id_list    = pd.DataFrame(id_list.values[1:], columns=headers)                             
        id_list    = id_list[['id']].dropna().values.tolist()

        # Get the dates and format them
        if satellite == 'Sentinel-2':
            date_list = [x[0].split('_')[1].split('T')[0] for x in id_list] #FOR SENTINEL 2 AJT
            date_list = [datetime.strptime(x,'%Y%m%d').date().isoformat() for x in date_list]
        else:
            date_list = [x[0].split('_')[2] for x in id_list] #FOR LANDSAT 8
            date_list = [datetime.strptime(x,'%Y%m%d').date().isoformat() for x in date_list]


    for i in range(img_collection.size().getInfo()):
        #try:
        #    print('>  ==> Datetime now: ' + str(id_list[i]) + '  '+ str(date_list[i]))
        #except:
        #    print('>  ==> Datetime NA')
        #    id_list.append(None)
        #    date_list.append(None)
        #    pass

        try:
            currentimg = ee.Image(collectionList.get(i))
            imgdate = datetime(1970, 1, 1, 0, 0) + timedelta(seconds=currentimg.date().getInfo()['value']/1000)
        
            try:
                wind_collection = ee.ImageCollection("ECMWF/ERA5/DAILY").filterDate(imgdate.strftime('%Y-%m-%d')).filterBounds(era5_region).select(['u_component_of_wind_10m','v_component_of_wind_10m'])
                wind = wind_collection.first()
                u = geemap.ee_to_numpy(wind.select('u_component_of_wind_10m'), region = era5_region)
                v = geemap.ee_to_numpy(wind.select('v_component_of_wind_10m'), region = era5_region)
                u = np.nanmean(u)
                v = np.nanmean(v)
            except:
                u10m.append(None)
                v10m.append(None)
                pass
            else:
                u10m.append(u)
                v10m.append(v)


            _SZA = 90 - currentimg.get('SUN_ELEVATION').getInfo()
            SZA.append(_SZA)
            
            lons = currentimg.pixelLonLat().select('longitude').reproject(crs=ee.Projection('EPSG:3395'), scale=30)
            lats = currentimg.pixelLonLat().select('latitude').reproject(crs=ee.Projection('EPSG:3395'), scale=30)
            lons = np.squeeze(geemap.ee_to_numpy(lons, region=region))
            lats = np.squeeze(geemap.ee_to_numpy(lats, region=region))


            B6channel = currentimg.select(swir1band).multiply(scaleFac)
            B7channel = currentimg.select(swir2band).multiply(scaleFac)
            SWIR1img = B6channel.reproject(crs=ee.Projection('EPSG:3395'), scale=30)
            SWIR2img = B7channel.reproject(crs=ee.Projection('EPSG:3395'), scale=30)

            # To numpy array
            SWIR1_geemap = geemap.ee_to_numpy(SWIR1img, region=region, default_value=_default_value).astype(float)
            SWIR2_geemap = geemap.ee_to_numpy(SWIR2img, region=region, default_value=_default_value).astype(float)
            # print(type(SWIR1_geemap))
            # print(np.where(np.isnan(SWIR1_geemap)))
            # print(np.where(SWIR1_geemap == _default_value))
            #print('None check: ', np.any(SWIR1_geemap is None))
            #print(type(SWIR1_geemap))
            #print(np.where(SWIR1_geemap == _default_value))
            #np.save('temp_geemap.npy', SWIR1_geemap)
            if np.any(SWIR1_geemap == _default_value):
                SWIR1_geemap[np.where(SWIR1_geemap == _default_value)] = np.nan
            if np.any(SWIR2_geemap == _default_value):
                SWIR2_geemap[np.where(SWIR2_geemap == _default_value)] = np.nan

            SWIR1_flat = np.reshape(np.squeeze(SWIR1_geemap),-1)
            SWIR2_flat = np.reshape(np.squeeze(SWIR2_geemap),-1)

            mask = np.where(np.logical_and(~np.isnan(SWIR1_flat), ~np.isnan(SWIR2_flat)))
            SWIR1_flat = SWIR1_flat[mask]
            SWIR2_flat = SWIR2_flat[mask]

            SWIR1_flat = np.array(SWIR1_flat).reshape(-1,1)

            model = HuberRegressor().fit(SWIR1_flat, SWIR2_flat)
            b0 = 1/model.coef_[0] #This slope is SWIR2/SWIR1 

            dR = ee.Image(B6channel.multiply((b0)).subtract(B7channel).divide(B7channel)).rename('dR')
            dR = dR.reproject(crs=ee.Projection('EPSG:3395'), scale=30)
            dR = np.squeeze(geemap.ee_to_numpy(dR, region=region, default_value=_default_value).astype(float))
            dR[np.where(dR == _default_value)] = np.nan

            if do_retrieval:
                test_retrieval = retrieve(dR, 'L8', method, targheight, obsheight, solarangle, obsangle, num_layers) ### retrieval
                z = test_retrieval*-1

            # get RGB, NIR, SWIRI, SWIRII channels from Landsat 8
            bchannel = currentimg.select(blueband).multiply(scaleFac).reproject(crs=ee.Projection('EPSG:3395'), scale=30)
            bchannel = np.squeeze(geemap.ee_to_numpy(bchannel, region=region, default_value=_default_value).astype(float))

            gchannel = currentimg.select(greenband).multiply(scaleFac).reproject(crs=ee.Projection('EPSG:3395'), scale=30)
            gchannel = np.squeeze(geemap.ee_to_numpy(gchannel, region=region, default_value=_default_value).astype(float))

            rchannel = currentimg.select(redband).multiply(scaleFac).reproject(crs=ee.Projection('EPSG:3395'), scale=30)
            rchannel = np.squeeze(geemap.ee_to_numpy(rchannel, region=region, default_value=_default_value).astype(float))

            nirchannel = currentimg.select(nirband).multiply(scaleFac).reproject(crs=ee.Projection('EPSG:3395'), scale=30)
            nirchannel = np.squeeze(geemap.ee_to_numpy(nirchannel, region=region, default_value=_default_value).astype(float))

            swir1channel = currentimg.select(swir1band).multiply(scaleFac).reproject(crs=ee.Projection('EPSG:3395'), scale=30)
            swir1channel = np.squeeze(geemap.ee_to_numpy(swir1channel, region=region, default_value=_default_value).astype(float))

            swir2channel = currentimg.select(swir2band).multiply(scaleFac).reproject(crs=ee.Projection('EPSG:3395'), scale=30)
            swir2channel = np.squeeze(geemap.ee_to_numpy(swir2channel, region=region, default_value=_default_value).astype(float))

            if satellite == 'Sentinel-2':
                cloudscore = currentimg.select('cloud_prob').reproject(crs=ee.Projection('EPSG:3395'), scale=30)
                cloudscore = np.squeeze(geemap.ee_to_numpy(cloudscore, region=region, default_value=None).astype(float))
            else:
                cloudscore = ee.Algorithms.Landsat.simpleCloudScore(currentimg).select(['cloud'])
                cloudscore = cloudscore.reproject(crs=ee.Projection('EPSG:3395'), scale=30).float()
                cloudscore = np.squeeze(geemap.ee_to_numpy(cloudscore, region=region, default_value=999)).astype(float)

            # Make sure the cloudscore isn't empty
            if cloudscore.all() == None:
                cloudscore    = np.empty(bchannel.shape)
                cloudscore[:] = 100

            ## Only do this for Landsat 7...
            #rchannel = filter_nans(rchannel, lons, lats, _default_value)
            #gchannel = filter_nans(gchannel, lons, lats, _default_value)
            #bchannel = filter_nans(bchannel, lons, lats, _default_value)
            #nirchannel = filter_nans(nirchannel, lons, lats, _default_value)
            #swir1channel = filter_nans(swir1channel, lons, lats, _default_value)
            #swir2channel = filter_nans(swir2channel, lons, lats, _default_value)
            #cloudscore = filter_nans(cloudscore, lons, lats, _default_value)
            #dR = filter_nans(dR, lons, lats, _default_value)

            #print(cloudscore.shape, swir1channel.shape, swir2channel.shape, nirchannel.shape, dR.shape, rchannel.shape,
            #      gchannel.shape, bchannel.shape)
            chanls = np.stack([dR, rchannel, gchannel, bchannel, nirchannel, swir1channel, swir2channel, cloudscore], axis=-1)
            date_list2.append(imgdate)

            if chanlarr is None:     # Initialize arrays
                # dRarr = dR[np.newaxis, :, :]
                chanlarr = chanls[np.newaxis, :, :, :]
                lonarr = lons[np.newaxis, :, :]
                latarr = lats[np.newaxis, :, :]
                if do_retrieval:
                    zarr = z[np.newaxis, :, :]
                else:
                    zarr = np.array([None])
            else:
                # dRarr = np.concatenate((dRarr, dR[np.newaxis, :, :]), axis=0)
                chanlarr = np.concatenate((chanlarr, chanls[np.newaxis, :, :, :]), axis=0)
                lonarr = np.concatenate((lonarr, lons[np.newaxis, :, :]), axis=0)
                latarr = np.concatenate((latarr, lats[np.newaxis, :, :]), axis=0)
                if do_retrieval:
                    zarr = np.concatenate((zarr, z[np.newaxis, :, :]), axis=0)
                else:
                    zarr = np.append(zarr, None)
        except Exception as e:
            print(">  ==> !!!Something went wrong!!!: " + str(e))
            # print(traceback.print_exc())
            # raise Exception
    return id_list, date_list, date_list2, chanlarr, zarr, lonarr, latarr, u10m, v10m, SZA


def interp2d(_arr, _kind='linear'):
    """
    arr: (W, H)
    """
    arr = _arr.copy()
    

    width = arr.shape[0]
    height = arr.shape[1]
    outarr = np.zeros((128, 128))
    
    xin, yin = np.linspace(1, 128, height), np.linspace(1, 128, width)
    gridxin, gridyin = np.meshgrid(xin, yin)
    xy = np.transpose(np.array([np.reshape(gridxin,np.prod(gridxin.shape)),np.reshape(gridyin,np.prod(gridyin.shape))]))
    gridxout, gridyout = np.meshgrid(np.linspace(1, 128, 128), np.linspace(1, 128, 128), indexing='xy')
    
    outarr[:, :] = griddata(xy, np.reshape(arr, np.prod(arr.shape)), (gridxout, gridyout), method='nearest')
    return outarr

def zstandard(arr):
    '''
    h, w
    '''
    _mu = np.nanmean(arr)
    _std = np.nanstd(arr)
    
    return (arr - _mu)/_std





def find_bg(lon, lat, imgdate, dR, _timewindow=180, _satellite='Landsat 5'):
    
    startdt = imgdate - timedelta(days=_timewindow)
    enddt = imgdate + timedelta(days=_timewindow)
    imgdate = datetime(year=imgdate.year, month=imgdate.month, day=imgdate.day)
    print(imgdate, startdt, enddt)
    
    img_id_list, date_list, img_date_list, imgchannels, imgxch4, imglons, imglats, u10m, v10m, SZA = get_plume(lon, 
                                                                                                          lat, 
                                                                                                          startdt.strftime('%Y-%m-%d'), 
                                                                                                          enddt.strftime('%Y-%m-%d'), 
                                                                                                          1.5, 1.5, 
                                                                                                          do_retrieval=False, 
                                                                                                          satellite=_satellite)
    
    dR = interp2d(dR)
    img_date_list = [datetime(year=s.year, month=s.month, day=s.day) for s in img_date_list]
    img_date_list = np.array(img_date_list)
    # print(type(img_date_list))
    # print(imgdate, img_date_list, img_date_list != imgdate)
    bgcandidates = img_date_list != imgdate
    bgcandidates2 = np.nanmin(imgchannels[:, :, :, 0], axis=(1, 2)) > -1
    bgcandidates = np.where(np.logical_and(bgcandidates, bgcandidates2))[0]
    # print(bgcandidates)
    imgchannels = np.take(imgchannels, bgcandidates, axis=0)
    img_date_list = np.take(img_date_list, bgcandidates)
    ssimlist = []
    bgraw = []
    
    for i in range(len(bgcandidates)):
        _dR = interp2d(imgchannels[i, :, :, 0])
        # ssimlist.append(ssim(zstandard(dR), zstandard(_dR)))
        ssimlist.append(ssim(dR, _dR, data_range=2))
        bgraw.append(_dR)
    
    ssimlist = np.array(ssimlist)
    bgraw = np.array(bgraw)
    # print(ssimlist)
    
    if len(ssimlist) == 0:
        return None, None, None
    else:
        candidatesize = np.sum(ssimlist >= 0.5)
        if candidatesize == 0:
            # only use best bg
            bgraw = np.take(bgraw, np.nanargmax(ssimlist), axis=0)
            
        # elif candidatesize <= 8:
        #     # average all candidates
        #     bgcandidates = np.where(ssimlist >= 0.1)
        #     bgraw = np.take(bgraw, bgcandidates, axis=0)
        #     bgraw = np.nanmean(bgraw, axis=0)
        # else:
        #     # only avereage top 8
        #     bgcandidates = np.argsort(ssimlist)[-8:]
        #     bgraw = np.take(bgraw, bgcandidates, axis=0)
        #     bgraw = np.nanmean(bgraw, axis=0)
        else:
            bgcandidates = np.where(ssimlist >= 0.5)
            bgraw = np.take(bgraw, bgcandidates, axis=0)[0]
            bgraw = np.nanmean(bgraw, axis=0)
            # bgraw = np.take(bgraw, np.nanargmax(ssimlist), axis=0)

        
        # model = HuberRegressor().fit(dR.flatten().reshape(-1, 1), bgraw.flatten().reshape(-1, 1))
        # b0 = model.coef_[0] #This slope is bg/dr   
        # print('ratio', b0)
        # bgraw = bgraw/b0
        # bgraw = bgraw + np.nanmean(dR - bgraw)
        dRmbg = dR - bgraw
        return bgraw, ssimlist, dR
    
def transform_raw_image(img_128):
    '''
    F, H, W
    '''
    # dR, red, green, blue, nir, swir1, swir2, cloudscore, bg
    
    red = img_128[:, :, 1]
    green = img_128[:, :, 2]
    blue = img_128[:, :, 3]
    nir = img_128[:, :, 4]

    dr = img_128[:, :, 0]
    dr[dr < -1] = -1
    dr = zstandard(dr)

    bg = img_128[:, :, 8]
    bg[bg<-1] = -1
    bg = -zstandard(bg)

    ndvi = (nir - red)/(nir + red)
    gs = 0.3*red + 0.59*green + 0.11*blue
    gs = zstandard(gs)
    ndvi = -zstandard(ndvi)


    model = HuberRegressor().fit(img_128[:, :, 0].flatten().reshape(-1, 1), img_128[:, :, 8].flatten().reshape(-1, 1))
    b0 = model.coef_[0] #This slope is dR/bg        
    zdrmbg1 = zstandard( (img_128[:, :, 0] - img_128[:, :, 8]*b0) )
    zdrmbg1 = np.clip(zdrmbg1, -4, 4)

    zdrmbg2 = zstandard(img_128[:, :, 0] - img_128[:, :, 8])
    
    print(dr.shape, bg.shape, zdrmbg1.shape, ndvi.shape, gs.shape)
    postimg = np.stack([dr, bg, zdrmbg1, zdrmbg2, ndvi, gs])
    postimg = postimg[np.newaxis, :, :, :]
    print(postimg.shape)
    return postimg
















def pred_plume_mask(inputs_128, pnet_models_weights):
    _inputs_128 = torch.tensor(inputs_128.astype(np.float32))
    maskpred = np.zeros((128, 128, len(pnet_models_weights)))

    for i in range(len(pnet_models_weights)):
        pnet_model.load_state_dict(torch.load(pnet_models_weights[i], map_location=torch.device('cpu')))    # torch.load('model_weights/PlumeNet_0_weights.h5'))
        _ = pnet_model.to(device)
        masknow = pnet_model(_inputs_128.to(device))
        masknow = masknow.detach().cpu().numpy()[0,0,:,:]
        masknow[masknow > 0.1] = 1
        masknow[masknow <= 0.1] = 0
        maskpred[:, :, i] = masknow

    meanmask = np.nanmean(maskpred, axis=-1)
    
    return meanmask


def delta_q(Q, ueff, L, domega, plume_area):
    dueff = 0.66
    dl = 0.1*L
    deltaq = (Q*dueff/ueff)**2 + (Q*dl/L)**2 + (ueff*domega/L*plume_area)**2
    deltaq = np.sqrt(deltaq)
    
    return deltaq



def plot_plume(lon, lat, yy, mm, dd, _satellite='Landsat 5', _timewindow=180):
    print('check')
    start_time = datetime(yy, mm, dd, 0, 0) # 1989-2-8, 1989-7-2, 
    end_time = start_time + timedelta(days=1)
    lonnow = lon
    latnow = lat
    img_id_list, date_list, img_date_list, imgchannels, imgxch4, imglons, imglats, u10m, v10m, SZA = get_plume(lonnow, 
                                                                                                          latnow, 
                                                                                                          start_time.strftime('%Y-%m-%d'), 
                                                                                                          end_time.strftime('%Y-%m-%d'), 
                                                                                                          1.5, 1.5, 
                                                                                                          do_retrieval=False, 
                                                                                                          satellite=_satellite)
    
    goodind = np.where(~np.any(np.isnan(imgchannels[:, :, :, 0]), axis=(1, 2)))[0][0]
    SZA = SZA[goodind]
    
    ratio1 = imgchannels[goodind, :, :, 0].shape[0]/128
    ratio2 = imgchannels[goodind, :, :, 0].shape[1]/128
    pixelsize = (30*ratio1)*(30*ratio2) # m^2
    #### look for background
    bg, drmbg, dr = find_bg(lonnow, latnow, img_date_list[goodind], imgchannels[goodind, :, :, 0], _timewindow, _satellite)
    
    img_128 = np.zeros((128, 128, 9))
    for i in range(8):
        img_128[:, :, i] = interp2d(imgchannels[goodind, :, :, i])
        
    bg = bg + np.nanmean(img_128[:, :, 0] - bg)
    
    img_128[:, :, 8] = bg
    inputs_128 = transform_raw_image(img_128)
    
    meanmask = pred_plume_mask(inputs_128, pnet_models_weights)
    
    _0p1lvl= meanmask.copy()
    _0p1lvl[_0p1lvl >= 0.1] = 1
    _0p1lvl[_0p1lvl < 0.1] = 0

    _0p5lvl = meanmask.copy()
    _0p5lvl[_0p5lvl >= 0.75] = 1
    _0p5lvl[_0p5lvl < 0.75] = 0
    
    #### retrieval
    num_layers = 100
    targheight = 0
    obsheight  = 100
    solarangle = SZA
    obsangle   = 0
    instrument = satellite_database[_satellite]['Instrument']
    method     = 'MBSP'
    ch4 = -rt.retrieve(dr, instrument, method, targheight, obsheight, solarangle, obsangle, num_layers=num_layers)
    ch4bg = -rt.retrieve(bg, instrument, method, targheight, obsheight, solarangle, obsangle, num_layers=num_layers)
    
    #### prepare figure
    fig, axs = pplt.subplots(ncols=2, nrows=2, refwidth=3)
    im = axs[0, 0].matshow(dr, cmap='coolwarm')
    axs[0, 0].set_title('Target scene')
    vmin, vmax = im.get_clim()
    
    axs[0, 1].matshow(bg, cmap='coolwarm', vmin=vmin, vmax=vmax)
    axs[0, 1].set_title('Background scene, SSIM = %.2f'%ssim(bg, dr, data_range=2))

    # get rgb image
    rgb = img_128[:, :, 1:4]
    brightness = np.nanmean(np.sqrt(rgb[:,:,0]**2 + rgb[:,:,1]**2 + rgb[:,:,2]**2))
    rgb = rgb/brightness


    enhance = ch4-ch4bg
    _maskout = _0p1lvl.copy()
    _maskout[_0p1lvl == 1] = np.nan
    _maskout[_0p1lvl == 0] = 1
    bias = np.nanmean(enhance*_maskout)  # make sure mean enhancebend outside plume mask is zero
    domega = np.nanstd(enhance*_maskout)/np.sqrt(np.sum(~np.isnan(_maskout)))
    
    enhance = enhance - bias

    im = axs[1, 0].matshow(enhance, cmap='bwr', vmin=-0.8, vmax=0.8)
    _maskin = _0p1lvl.copy()
    axs[1, 0].contour(_maskin, levels=[0.5], colors='black', linestyles='dashed')
    _maskin[_maskin <= 0.1] = np.nan
    axs[1, 0].contour(_0p5lvl, levels=[0.5], colors='black', linestyles='solid')
    axs[1, 0].set_title('Retrieved CH4 enhancement')
    axs[1, 0].colorbar(im, loc='b', label='mol/m2')

    ratio = imgchannels[0, :, :, 0].shape[0]/128
    pixelsize = (30*ratio)**2 # m^2
    ch4kg = enhance*pixelsize*16.04/1000
    ime = ch4kg*_maskin
    
    try:
        ueffect = np.sqrt(u10m[0]**2 + v10m[0]**2)
        ueffect = 0.33*ueffect+0.45
        imesum = np.nansum(ime)
        plume_area = np.nansum(_maskin)*pixelsize
        LLL = np.sqrt(plume_area)
        QQQ = imesum*ueffect/LLL*3600.
        
        dQQQ = delta_q(QQQ, ueffect, LLL, domega, plume_area)
        
        axs[1, 1].text(0.1, 0.9, r'Q=%.2f$\pm$%.2f kg/hr'%(QQQ, dQQQ), transform=axs[1, 1].transAxes, fontsize=15)
    except:
        QQQ = None
        dQQQ = None
        axs[1, 1].text(0.1, 0.9, 'Q=NA', transform=axs[1, 1].transAxes, fontsize=15)

    im_temp = axs[1, 1].matshow(ime, cmap='inferno', vmin=0, vmax=8)
    rgb[np.where(~np.isnan(ime))] = im_temp.cmap(im_temp.norm(ime[np.where(~np.isnan(_maskin))]))[:, :3] # [255, 0, 0]
    im = axs[1, 1].imshow(rgb)
    axs[1, 1].colorbar(im_temp, loc='b', label='kg')

    fig.format(suptitleweight='bold', titlesize=12, yticklabels=[], xticklabels=[], abc=True, abcloc='ul', abcstyle='A', abcsize=15,
              suptitle='Lon: %.3f, Lat: %.3f, %s'%(lonnow, latnow, start_time.strftime('%Y-%m-%d')), suptitlesize=15)
    return fig, QQQ, dQQQ


def get_plume_info(lon, lat, yy, mm, dd, _satellite='Landsat 5', _timewindow=180):
    # print('check')
    start_time = datetime(yy, mm, dd, 0, 0) # 1989-2-8, 1989-7-2, 
    end_time = start_time + timedelta(days=1)
    lonnow = lon
    latnow = lat
    img_id_list, date_list, img_date_list, imgchannels, imgxch4, imglons, imglats, u10m, v10m, SZA = get_plume(lonnow, 
                                                                                                          latnow, 
                                                                                                          start_time.strftime('%Y-%m-%d'), 
                                                                                                          end_time.strftime('%Y-%m-%d'), 
                                                                                                          1.5, 1.5, 
                                                                                                          do_retrieval=False, 
                                                                                                          satellite=_satellite)
    
    goodind = np.where(~np.any(np.isnan(imgchannels[:, :, :, 0]), axis=(1, 2)))[0][0]
    SZA = SZA[goodind]
    
    ratio1 = imgchannels[goodind, :, :, 0].shape[0]/128
    ratio2 = imgchannels[goodind, :, :, 0].shape[1]/128
    pixelsize = (30*ratio1)*(30*ratio2) # m^2
    #### look for background
    bg, drmbg, dr = find_bg(lonnow, latnow, img_date_list[goodind], imgchannels[goodind, :, :, 0], _timewindow, _satellite)
    
    img_128 = np.zeros((128, 128, 9))
    for i in range(8):
        img_128[:, :, i] = interp2d(imgchannels[goodind, :, :, i])
        
    imglons_128 = interp2d(imglons[goodind, :, :])
    imglats_128 = interp2d(imglats[goodind, :, :])
        
    bg = bg + np.nanmean(img_128[:, :, 0] - bg)
    
    img_128[:, :, 8] = bg
    inputs_128 = transform_raw_image(img_128)
    
    meanmask = pred_plume_mask(inputs_128, pnet_models_weights)
    
    _0p1lvl= meanmask.copy()
    _0p1lvl[_0p1lvl >= 0.1] = 1
    _0p1lvl[_0p1lvl < 0.1] = 0

    _0p5lvl = meanmask.copy()
    _0p5lvl[_0p5lvl >= 0.75] = 1
    _0p5lvl[_0p5lvl < 0.75] = 0
    
    #### retrieval
    num_layers = 100
    targheight = 0
    obsheight  = 100
    solarangle = SZA
    obsangle   = 0
    instrument = satellite_database[_satellite]['Instrument']
    method     = 'MBSP'
    ch4 = -rt.retrieve(dr, instrument, method, targheight, obsheight, solarangle, obsangle, num_layers=num_layers)
    ch4bg = -rt.retrieve(bg, instrument, method, targheight, obsheight, solarangle, obsangle, num_layers=num_layers)
    
    # get rgb image
    rgb = img_128[:, :, 1:4]
    brightness = np.nanmean(np.sqrt(rgb[:,:,0]**2 + rgb[:,:,1]**2 + rgb[:,:,2]**2))
    rgb = rgb/brightness
    
    
    enhance = ch4-ch4bg
    _maskout = _0p1lvl.copy()
    _maskout[_0p1lvl == 1] = np.nan
    _maskout[_0p1lvl == 0] = 1
    bias = np.nanmean(enhance*_maskout)  # make sure mean enhancebend outside plume mask is zero
    domega = np.nanstd(enhance*_maskout)/np.sqrt(np.sum(~np.isnan(_maskout)))
    
    enhance = enhance - bias
    _maskin = _0p1lvl.copy()
    _maskin[_maskin <= 0.1] = np.nan

    ratio = imgchannels[0, :, :, 0].shape[0]/128
    pixelsize = (30*ratio)**2 # m^2
    ch4kg = enhance*pixelsize*16.04/1000
    ime = ch4kg*_maskin

    im_temp = plt.matshow(ime, cmap='inferno', vmin=0, vmax=8)
    rgb[np.where(~np.isnan(ime))] = im_temp.cmap(im_temp.norm(ime[np.where(~np.isnan(_maskin))]))[:, :3] # [255, 0, 0]
    
    try:
        ueffect = np.sqrt(u10m[0]**2 + v10m[0]**2)
        ueffect = 0.33*ueffect+0.45
        imesum = np.nansum(ime)
        plume_area = np.nansum(_maskin)*pixelsize
        LLL = np.sqrt(plume_area)
        QQQ = imesum*ueffect/LLL*3600.
        dQQQ = delta_q(QQQ, ueffect, LLL, domega, plume_area)
        

    except:
        QQQ = None
        dQQQ = None

    
    return imglons_128, imglats_128, dr, bg, enhance, rgb, _0p1lvl, _0p5lvl, QQQ, dQQQ, ime, u10m, v10m
