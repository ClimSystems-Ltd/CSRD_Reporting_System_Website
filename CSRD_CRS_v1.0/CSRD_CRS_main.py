"""export NUMEXPR_MAX_THREADS=5; streamlit run CSRD_CRS_main.py"""

# ---------------------------------------------------------------------------

# This is the ClimateHub Data Service (CDS v1.0) toolbox of the geospatial data science --- CSRD Reporting System v1.0
# '''
# Author: Zhaoquan YU - Johnny
# Create: 20/11/2022
# Last Modify: 10/06/2025
# streamlit                  1.40.1
# streamlit-authenticator    0.4.1
# '''
# conda install streamlit_folium streamlit_authenticator leafmap xarray geopandas folium shapely streamlit_js_eval numba scipy python-docx cartopy fsspec aiohttp xlsxwriter openpyxl seaborn

# ---------------------------------------------------------------------------

import base64
import calendar
import folium
import geopandas as gpd
import glob
import json
import leafmap.foliumap as leafmap
import numpy as np
import os 
import pandas as pd
import pickle
import re
import requests
import shapely
import shutil
import streamlit as st
import streamlit_authenticator as stauth
import streamlit_folium as stf
import time
import uuid
import xarray as xr
import yaml

from folium.plugins import Draw, MeasureControl
from fsspec.implementations.http import HTTPFileSystem
from io import BytesIO, StringIO
from matplotlib import cm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from scipy import spatial
from streamlit_js_eval import streamlit_js_eval
from zipfile import ZipFile

####################################################################################

# ----------------------------------------------------------------------------------------
# Global objects
# ----------------------------------------------------------------------------------------

fs = HTTPFileSystem()
cover_info_page = pd.DataFrame([
    ['Data Explainer', 'This document contains the raw and processed output for the site(s) selected. Please refer to the technical memo for more details and implications of the results.'],
    ['Data Disclosure',
f'''The Climate-Related Physical Risk Report: Warehouse was prepared by ClimSystems Ltd. The data and methods presented in this report represent ClimSystems' professional judgment based on information made available by the recipient of this report and are true and correct to the best of ClimSystems' knowledge as of the date of the report's development. ClimSystems does not independently verify the information provided during the development of this report. While ClimSystems have no reason to doubt the outputs of the information provided, however, the report is limited by the data applied in generating the report.

ClimSystems takes great care to ensure the climate information in this report is as correct and accurate as possible. The climate information provided is subject to the uncertainties of scientific and technical research; it may not be accurate, current or complete; be subject to change without notice and is not a substitute for independent professional advice. Users should obtain any appropriate professional advice relevant to their particular circumstances. ClimSystems does not guarantee the information provided and accepts no legal liability whatsoever arising from, or connected to, the use of any material contained therein.

Climate data and related risk analytics are, by their nature, dynamic and ongoing functions. Therefore, it is recommended that the recipient of this report routinely incorporate the latest climate data into all future adaptation planning.

ClimSystems recommends that the recipient of this report use their skill and care concerning their use of the climatic and risk information, and those users carefully evaluate the currency, completeness, and relevance of the material for their purposes. Be particularly mindful of analysis at the coastline, where boundaries between the marine and terrestrial environments may be inaccurate. Any use of the data is solely at the recipient's own risk. We understand that no party can rely upon the results of the assessments for design purposes.

To the extent permitted by law, ClimSystems makes no representation or warranty (expressed or implied) as to merchantability or performance of the data; about the fitness of the data for the permitted use; or that the data does not infringe the intellectual property rights or any other right of any person. The recipient indemnifies and releases ClimSystems against all claims, demands, suits, liability, loss, or expenses arising directly or indirectly from the recipient's use of the data or any breach of this agreement by the recipient. This report does not purport to give legal advice. Qualified legal advisors can only give this advice.

This data has been prepared exclusively for the recipient of this report for the location the recipient identified and may not be relied upon by any other person or entity without ClimSystems' express written permission.'''
    ]], columns=['', ' '])

# ----------------------------------------------------------------------------------------
# Global processing
# ----------------------------------------------------------------------------------------

def build_private_cart(private_cart_key, st_obj):
    with st_obj.popover(":shopping_trolley: My Cart", use_container_width=True):
        for i,key in enumerate(st.session_state[private_cart_key]):
            st.download_button(
                label=f'📥 {key}',
                data=st.session_state[private_cart_key][key],
                file_name=key,
                key=f"{time.strftime('%Y.%m.%d.%H.%M.%S',time.localtime(time.time()))}.{i}",
                )

def crop_spatial_data(_ds, _bbox, _mask):
    (bottom, top), (left, right) = sorted(_bbox[0:2]), sorted(_bbox[2:4])
    if _mask and _ds.ClimateHub in st.session_state['region_cookie']:
        _mask = _ds.copy()
        for vn in _ds.data_vars: _mask[vn].values = _ds[vn] * st.session_state['region_cookie'][_ds.ClimateHub]
        return _mask.sel(
            lat=slice(top, bottom) if _ds.lat[1]<_ds.lat[0] else slice(bottom, top),
            lon=slice(left, right) if _ds.lon[1]>_ds.lon[0] else slice(right, left)
            ).compute()
    else:
        return _ds.sel(
            lat=slice(top, bottom) if _ds.lat[1]<_ds.lat[0] else slice(bottom, top),
            lon=slice(left, right) if _ds.lon[1]>_ds.lon[0] else slice(right, left)
            ).compute()

def distance_coast(latlng, ne_10m_coastline, coastline):

    if ne_10m_coastline is None or coastline is None: return None
    get_nearest_distance_ne_10m = get_nearest_distance(ne_10m_coastline, latlng, k=5)
    r = np.ceil(abs(np.array(get_nearest_distance_ne_10m['nearestCoords(lat,lng)']) - latlng[0]).max()) * 2
    coastline = coastline[coastline[:,0]>=latlng[0][0]-r]
    coastline = coastline[coastline[:,0]<=latlng[0][0]+r]
    coastline = coastline[coastline[:,1]>=latlng[0][1]-r]
    coastline = coastline[coastline[:,1]<=latlng[0][1]+r]

    return {'queryCoords(lat,lng)': latlng[0], 'Distance_Coast': get_nearest_distance(coastline, latlng) if coastline.size>0 else {'kilometre': get_nearest_distance_ne_10m['kilometre'][0], 'nearestCoords(lat,lng)': get_nearest_distance_ne_10m['nearestCoords(lat,lng)'][0]}}

def get_country_continent(lat, lng, shp=None):
    shp = gpd.read_file('CRA_Parameter/countries.dbf') if shp is None else shp
    within = shp['geometry'].geometry.contains( gpd.points_from_xy([lng],[lat])[0] )
    return shp.iloc[within[within].index][['COUNTRY','CONTINENT']].values.tolist()

# ----------------------------------------------------------------------------------------
# Cache resource
# ----------------------------------------------------------------------------------------

@st.cache_resource
def load_cookie(cookie_path):
    return {fn.split(os.sep)[-1].split('_')[0]: raster2xr(fn).values for fn in glob.glob(f'{cookie_path}{os.sep}*_cookie.asc')}

@st.cache_resource
def load_excel(fn, sheet_name=None, index_col=None):
    if os.path.exists(fn): return pd.read_excel(fn, sheet_name=sheet_name, index_col=index_col)

@st.cache_resource
def load_csv(fn, index_col=None):
    if os.path.exists(fn): return pd.read_csv(fn, index_col=index_col)

@st.cache_resource
def load_shape_data(fn):
    if os.path.exists(fn): return gpd.read_file(fn)

@st.cache_resource
def load_DataArray(fn):
    if os.path.exists(fn): return xr.open_dataarray(fn)

@st.cache_resource
def load_Dataset(fns, concat_dim=None, engine=None, parallel=True):
    if len(fns)>0: return xr.open_mfdataset(fns, combine='nested' if concat_dim is not None else 'by_coords', concat_dim=concat_dim, engine=engine, parallel=parallel)

@st.cache_resource
def load_coast_data():
    if not (os.path.exists(f'CRA_Parameter{os.sep}ne_10m_coastline.shp') and os.path.exists(f'CRA_Parameter{os.sep}global_landline_coords.npy')): return [None,None]
    ne_10m_coastline = shape2coords(f'CRA_Parameter{os.sep}ne_10m_coastline.shp')
    coastline = np.load(f'CRA_Parameter{os.sep}global_landline_coords.npy')
    return ne_10m_coastline, coastline

@st.cache_resource
def load_lib_so(key, url='http://192.168.1.16:8888'):
    return requests.get(f'{url}/lib/{key}?key={YOUR_API_KEY}').json()[key]

@st.cache_resource
def merge_zarr(IP,vns):
    return xr.merge([ xr.open_zarr(fs.get_mapper(f"{IP}/datasets/{vn}/zarr/"), consolidated=True) for vn in vns ])

# ----------------------------------------------------------------------------------------
# Request and produce
# ----------------------------------------------------------------------------------------

@st.cache_data
def Request_API_v3(params, HOST_v3, index_list=["climate_zone","elevation","earthquake","cyclone","tsunami","hail","distance_coast","water_stress"]):
    r = requests.post(f'{HOST_v3}/api/v3/app/', json=params).json()
    info = {}
    if index_list:
        for i in range(len(params['lat'])):
            info[i] = requests.get(f"{HOST_v3}/api/v3/index/?key={YOUR_API_KEY}&nscore={params['nscore']}&latlng={params['lat'][i]},{params['lon'][i]}&index={'+'.join(index_list)}").json()
    r['Information'] = info
    return r

@st.dialog("Please wait ...")
def exporting(export_time, locate, params, HOST_v3, index_list, tiled_theme, rank, private_cart_key):

    params.update({'lat': locate['Latitude'].values.tolist(), 'lon': locate['Longitude'].values.tolist()})
    params.update({'define_baseline': {k:locate[k].values.tolist() for k in params['variable'] if k in locate.columns}})
    month_abbr  = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ofn_header  = f"Export_{export_time}"

    output_path = f"CRA_{params['email']}_{export_time}" # local path
    os.makedirs(output_path, exist_ok=True)

    r = Request_API_v3(params, HOST_v3, index_list=index_list)

    with st.status(f"Exporting: {len(locate)} locations data.", expanded=True) as status:

        st.write("Processing Data ...")
        # convert to DataSet ###################################################
        data_ds = xr.Dataset.from_dict(r['data']).assign_attrs(Information=str(r['Information'])).round(2)
        score_ds = xr.Dataset.from_dict(r['score']).assign_attrs(Information=str(r['Information']))
        change_ds = xr.Dataset.from_dict(r['change']).assign_attrs(Information=str(r['Information'])).round(2)
        if 'month' in data_ds:
            data_ds = data_ds.assign_coords({'month': month_abbr})
            change_ds = change_ds.assign_coords({'month': month_abbr})

        st.write("Getting Data Source ...")
        # save to CSV ###################################################
        pds_df = pd.DataFrame.from_dict(r['Point_Data_Source']).T
        pds_df.columns = [f'Data Source@{c}' for c in r['Request_Body']['variable'][:len(pds_df.columns)]]
        pds_df['SN'] = pds_df.index.astype(int)+1
        pds_df = pd.wide_to_long(pds_df, stubnames='Data Source', i='SN', j='Indicator', sep='@', suffix=r'\w+').sort_index()
        pds_df['API Source'] = HOST_v3
        pds_df.to_csv(f'{output_path}{os.sep}{ofn_header}_Point_Data_Source.csv')

        st.write("Generating Metadata ...")
        # save to CSV ###################################################
        meta_df = export_Metadata_from_data_source(data_ds, pds_df.reset_index(), meta_dict={'standard_name':'','units':'','change_method':'','change_units':'','spatial_ref':'EPSG:4326'})
        meta_df.to_csv(f'{output_path}{os.sep}{ofn_header}_Metadata.csv', index=False)

        st.write("Generating Readme ...")
        #copy to PDF ###################################################
        if os.name == 'nt':
            os.system(f'copy CRA_Parameter{os.sep}Data_Package_readme_template.pdf {output_path}{os.sep}{ofn_header}_Readme.pdf')
        else:
            os.system(f'cp CRA_Parameter{os.sep}Data_Package_readme_template.pdf {output_path}{os.sep}{ofn_header}_Readme.pdf')

        st.write("Generating netCDFs ...")
        # save to netCDF ###################################################
        data_ds.to_netcdf(f'{output_path}{os.sep}{ofn_header}_Data.nc')
        score_ds.to_netcdf(f'{output_path}{os.sep}{ofn_header}_Score.nc')
        change_ds.to_netcdf(f'{output_path}{os.sep}{ofn_header}_Change.nc')

        st.write(f"Converting Format ...")
        # convert to DataFrame ###################################################
        index_df = pd.DataFrame.from_dict(eval(data_ds.Information)).T
        columns = []
        if 'Elevation' in index_df.columns:
            index_df['Elevation'] = index_df['Elevation'].apply(
            lambda i: list(i.values())[0]) 
            columns.append('Elevation')   
        if 'KG_Climate_Classification' in index_df.columns:
            index_df['KG_Climate_Classification'] = index_df['KG_Climate_Classification'].apply(
            lambda i: list(list(i.values())[0].values())[0][1])
            columns.append('KG_Climate_Classification')   
        if 'Earthquake_Intensity' in index_df.columns:
            index_df['Earthquake_Intensity'] = index_df['Earthquake_Intensity'].apply(
            lambda i: int(list(i.keys())[0]) + 1)
            columns.append('Earthquake_Intensity')   
        if 'Cyclone_Intensity' in index_df.columns:
            index_df['Cyclone_Intensity'] = index_df['Cyclone_Intensity'].apply(
            lambda i: list(i.values())[0][0].split(': ')[-1])
            columns.append('Cyclone_Intensity')   
        if 'Cyclone_Frequency' in index_df.columns:
            index_df['Cyclone_Frequency'] = index_df['Cyclone_Frequency'].apply(
            lambda i: list(i.values())[0][1])
            columns.append('Cyclone_Frequency')   
        if 'Tsunami' in index_df.columns:
            index_df['Tsunami_Hazard_Score'] = index_df['Tsunami'].apply(
            lambda i: list(i.values())[0])
            columns.append('Tsunami_Hazard_Score')   
            index_df['Tsunami_Maximum_Inundation_Height'] = index_df['Tsunami'].apply(
            lambda i: list(i.values())[1])
            columns.append('Tsunami_Maximum_Inundation_Height')   
        if 'Hail' in index_df.columns:
            index_df['Hail'] = index_df['Hail'].apply(
            lambda i: list(i.values())[0])
            columns.append('Hail')   
        if 'Distance_Coast' in index_df.columns:
            index_df['Distance_Coast'] = index_df['Distance_Coast'].apply(
            lambda i: list(i.values())[0])
            columns.append('Distance_Coast')   
        if 'Water_Stress' in index_df.columns:
            index_df['Water_Stress'] = index_df['Water_Stress'].apply(
            lambda i: list(i.values())[0])
            columns.append('Water_Stress')   
        index_df = index_df[columns].rename(
            columns={
            'KG_Climate_Classification': 'Climate Zone',
            'Elevation': 'Elevation(m)',
            'Earthquake_Intensity': 'Earthquake Intensity(1-10)',
            'Cyclone_Intensity': 'Cyclone Intensity(I-V)',
            'Cyclone_Frequency': 'Cyclone Frequency',
            'Tsunami_Hazard_Score': 'Tsunami Hazard Score(1-5)',
            'Tsunami_Maximum_Inundation_Height': 'Tsunami Maximum Inundation Height(m)',
            'Hail': 'Hail Frequency(times/year)',
            'Distance_Coast': 'Distance Coast(km)',
            'Water_Stress': 'Water Stress Category',
            })
        index_df.insert(0, 'Longitude', locate['Longitude'].values)
        index_df.insert(0, 'Latitude', locate['Latitude'].values)

        if 'Address' in locate.columns:
            index_df.insert(0, 'Address', locate['Address'].values)
            data_ds = data_ds.assign_coords({'point': [f'{i+1}<-split->{v}' for i,v in enumerate(locate['Address'].values)]})
            score_ds = score_ds.assign_coords({'point': [f'{i+1}<-split->{v}' for i,v in enumerate(locate['Address'].values)]})
            change_ds = change_ds.assign_coords({'point': [f'{i+1}<-split->{v}' for i,v in enumerate(locate['Address'].values)]})
        else:
            data_ds = data_ds.assign_coords({'point': [f'{i+1}<-split->-N/A-' for i in locate.index]})
            score_ds = score_ds.assign_coords({'point': [f'{i+1}<-split->-N/A-' for i in locate.index]})
            change_ds = change_ds.assign_coords({'point': [f'{i+1}<-split->-N/A-' for i in locate.index]})
        index_df.insert(0, 'SN', [str(i+1) for i in locate.index])

        for vn in data_ds.data_vars:
            data_ds = data_ds.rename({vn: f'{vn} ({data_ds[vn].units})'})
            score_ds = score_ds.rename({vn: f'{vn} ({score_ds[vn].units})'})
            change_ds = change_ds.rename({vn: f'{vn} ({change_ds[vn].units})'})
        data_df = convert_df(data_ds, 'Value').merge(index_df[['SN','Latitude','Longitude']], how='left', on='SN')
        score_df = convert_df(score_ds, 'Score').replace(0, '-N/A-').merge(index_df[['SN','Latitude','Longitude']], how='left', on='SN')
        change_df = convert_df(change_ds, 'Change').merge(index_df[['SN','Latitude','Longitude']], how='left', on='SN')

        try:
            st.write(f"Generating Excel ...")
            # save to Excel ###################################################
            Excel_io = convert_excel(index_df, data_df, score_df, change_df, tiled_theme, rank=rank)
            with open(f'{output_path}{os.sep}{ofn_header}_Data_Score_Change.xlsx', 'wb') as f:
                f.write(Excel_io.getbuffer())
        except:
            st.write(f"Generating CSVs ...")
            # save to CSVs ###################################################
            convert_csv(index_df, data_df, score_df, change_df, f'{output_path}{os.sep}{ofn_header}', tiled_theme, rank=rank, only_files=True)

        st.write(f"Compressing Data ...")
        Folder_io = BytesIO()
        with ZipFile(Folder_io, 'w') as archive:
            for fn in glob.glob(f'{output_path}{os.sep}*.*'):
                archive.write(fn, arcname=fn.split(os.sep)[-1])
                os.remove(fn)
            os.rmdir(output_path)

        key = f'Export_{export_time}.zip'
        st.session_state[private_cart_key][key] = Folder_io

        st.download_button(
            label=f'📥 {key}',
            data=Folder_io,
            file_name=key,
            on_click=st.stop,
            key=time.strftime('%Y.%m.%d.%H.%M.%S',time.localtime(time.time())),
            )
        status.update(label="Export completed to download.:point_down:", state="complete", expanded=True)

@st.dialog("Please wait ...")
def reporting(build_time, locate, report_type, params, HOST_v3, years, flood_scheme_path, tiled_theme, rank, private_cart_key):

    params.update({'lat': locate['Latitude'].values.tolist(), 'lon': locate['Longitude'].values.tolist()})
    params.update({'define_baseline': {k:locate[k].values.tolist() for k in params['variable'] if k in locate.columns}})
    month_abbr  = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ofn_header  = f'Export_{build_time}'

    output_path = Path(f"//AsustorFour/Home/CRA_{params['email']}_{build_time}").format() # network path on Windows
    # output_path = f"CRA_{params['email']}_{build_time}" # local path
    os.makedirs(output_path, exist_ok=True)

    r = Request_API_v3(params, HOST_v3, index_list=api_index[HOST_v3])

    with st.status(f"Building: {len(locate)} reports about {len(locate)*0.85} minutes.", expanded=True) as status:

        st.write("Processing Data ...")
        # change source data with Modified_data #################################
        if (os.path.exists(f'Modified_data{os.sep}_Data.nc') and
            os.path.exists(f'Modified_data{os.sep}_Score.nc') and
            os.path.exists(f'Modified_data{os.sep}_Change.nc')
            ):
            data_ds = xr.open_dataset(f'Modified_data{os.sep}_Data.nc')
            score_ds = xr.open_dataset(f'Modified_data{os.sep}_Score.nc')
            change_ds = xr.open_dataset(f'Modified_data{os.sep}_Change.nc')
        else:
            pds_df = pd.DataFrame.from_dict(r['Point_Data_Source']).T
            pds_df.columns = [f'Data Source@{c}' for c in r['Request_Body']['variable'][:len(pds_df.columns)]]
            pds_df['SN'] = pds_df.index.astype(int)+1
            pds_df = pd.wide_to_long(pds_df, stubnames='Data Source', i='SN', j='Indicator', sep='@', suffix=r'\w+').sort_index()
            pds_df['API Source'] = HOST_v3
            pds_df.to_csv(f'{output_path}{os.sep}{ofn_header}_Point_Data_Source.csv')

            # convert to DataSet ###################################################
            data_ds = xr.Dataset.from_dict(r['data']).assign_attrs(Information=str(r['Information'])).round(2)
            score_ds = xr.Dataset.from_dict(r['score']).assign_attrs(Information=str(r['Information']))
            change_ds = xr.Dataset.from_dict(r['change']).assign_attrs(Information=str(r['Information'])).round(2)

            if 'month' in data_ds:
                data_ds = data_ds.assign_coords({'month': month_abbr})
                change_ds = change_ds.assign_coords({'month': month_abbr})

            st.write("Generating Metadata ...")
            # save to CSV ###################################################
            meta_df = export_Metadata_from_data_source(data_ds, pds_df.reset_index(), meta_dict={'standard_name':'','units':'','change_method':'','change_units':'','spatial_ref':'EPSG:4326'})
            meta_df.to_csv(f'{output_path}{os.sep}{ofn_header}_Metadata.csv', index=False)

            st.write("Generating Readme ...")
            #copy to PDF ###################################################
            if os.name == 'nt':
                os.system(f'copy CRA_Parameter{os.sep}Report_Package_readme_template.pdf {output_path}{os.sep}{ofn_header}_Readme.pdf')
            else:
                os.system(f'cp CRA_Parameter{os.sep}Report_Package_readme_template.pdf {output_path}{os.sep}{ofn_header}_Readme.pdf')

        # save to netCDF ###################################################
        data_ds.to_netcdf(f'{output_path}{os.sep}{ofn_header}_Data.nc')
        score_ds.to_netcdf(f'{output_path}{os.sep}{ofn_header}_Score.nc')
        change_ds.to_netcdf(f'{output_path}{os.sep}{ofn_header}_Change.nc')

        st.write("Generating Reports ...")
        print(f'{report_type}: {ofn_header}{os.linesep}{locate}')
        Jr_response = report_engine(report_type, output_path, locate, years, flood_scheme_path)
        if isinstance(Jr_response, dict):
            st.warning(Jr_response)
            status.update(label="Build failure!", state="error", expanded=True)
            for fn in glob.glob(f'{output_path}{os.sep}*.*'):
                os.remove(fn)
            os.rmdir(output_path)
            return None

        st.write(f"Converting Format ...")
        # convert to DataFrame ###################################################
        index_df = pd.DataFrame.from_dict(eval(data_ds.Information)).T
        columns = []
        if 'Elevation' in index_df.columns:
            index_df['Elevation'] = index_df['Elevation'].apply(
            lambda i: list(i.values())[0]) 
            columns.append('Elevation')   
        if 'KG_Climate_Classification' in index_df.columns:
            index_df['KG_Climate_Classification'] = index_df['KG_Climate_Classification'].apply(
            lambda i: list(list(i.values())[0].values())[0][1])
            columns.append('KG_Climate_Classification')   
        if 'Earthquake_Intensity' in index_df.columns:
            index_df['Earthquake_Intensity'] = index_df['Earthquake_Intensity'].apply(
            lambda i: int(list(i.keys())[0]) + 1)
            columns.append('Earthquake_Intensity')   
        if 'Cyclone_Intensity' in index_df.columns:
            index_df['Cyclone_Intensity'] = index_df['Cyclone_Intensity'].apply(
            lambda i: list(i.values())[0][0].split(': ')[-1])
            columns.append('Cyclone_Intensity')   
        if 'Cyclone_Frequency' in index_df.columns:
            index_df['Cyclone_Frequency'] = index_df['Cyclone_Frequency'].apply(
            lambda i: list(i.values())[0][1])
            columns.append('Cyclone_Frequency')   
        if 'Tsunami' in index_df.columns:
            index_df['Tsunami_Hazard_Score'] = index_df['Tsunami'].apply(
            lambda i: list(i.values())[0])
            columns.append('Tsunami_Hazard_Score')   
            index_df['Tsunami_Maximum_Inundation_Height'] = index_df['Tsunami'].apply(
            lambda i: list(i.values())[1])
            columns.append('Tsunami_Maximum_Inundation_Height')   
        if 'Hail' in index_df.columns:
            index_df['Hail'] = index_df['Hail'].apply(
            lambda i: list(i.values())[0])
            columns.append('Hail')   
        if 'Distance_Coast' in index_df.columns:
            index_df['Distance_Coast'] = index_df['Distance_Coast'].apply(
            lambda i: list(i.values())[0])
            columns.append('Distance_Coast')   
        if 'Water_Stress' in index_df.columns:
            index_df['Water_Stress'] = index_df['Water_Stress'].apply(
            lambda i: list(i.values())[0])
            columns.append('Water_Stress')   
        index_df = index_df[columns].rename(
            columns={
            'KG_Climate_Classification': 'Climate Zone',
            'Elevation': 'Elevation(m)',
            'Earthquake_Intensity': 'Earthquake Intensity(1-10)',
            'Cyclone_Intensity': 'Cyclone Intensity(I-V)',
            'Cyclone_Frequency': 'Cyclone Frequency',
            'Tsunami_Hazard_Score': 'Tsunami Hazard Score(1-5)',
            'Tsunami_Maximum_Inundation_Height': 'Tsunami Maximum Inundation Height(m)',
            'Hail': 'Hail Frequency(times/year)',
            'Distance_Coast': 'Distance Coast(km)',
            'Water_Stress': 'Water Stress Category',
            })
        index_df.insert(0, 'Longitude', locate['Longitude'].values)
        index_df.insert(0, 'Latitude', locate['Latitude'].values)
        if 'Address' in locate.columns:
            index_df.insert(0, 'Address', locate['Address'].values)
            data_ds = data_ds.assign_coords({'point': [f'{i+1}<-split->{v}' for i,v in enumerate(locate['Address'].values)]})
            score_ds = score_ds.assign_coords({'point': [f'{i+1}<-split->{v}' for i,v in enumerate(locate['Address'].values)]})
            change_ds = change_ds.assign_coords({'point': [f'{i+1}<-split->{v}' for i,v in enumerate(locate['Address'].values)]})
        else:
            data_ds = data_ds.assign_coords({'point': [f'{i+1}<-split->-N/A-' for i in locate.index]})
            score_ds = score_ds.assign_coords({'point': [f'{i+1}<-split->-N/A-' for i in locate.index]})
            change_ds = change_ds.assign_coords({'point': [f'{i+1}<-split->-N/A-' for i in locate.index]})
        index_df.insert(0, 'SN', [str(i+1) for i in locate.index])
        # convert to DataFrame ###################################################
        for vn in data_ds.data_vars:
            data_ds = data_ds.rename({vn: f'{vn} ({data_ds[vn].units})'})
            score_ds = score_ds.rename({vn: f'{vn} ({score_ds[vn].units})'})
            change_ds = change_ds.rename({vn: f'{vn} ({change_ds[vn].units})'})
        data_df = convert_df(data_ds, 'Value').merge(index_df[['SN','Latitude','Longitude']], how='left', on='SN')
        score_df = convert_df(score_ds, 'Score').replace(0, '-N/A-').merge(index_df[['SN','Latitude','Longitude']], how='left', on='SN')
        change_df = convert_df(change_ds, 'Change').merge(index_df[['SN','Latitude','Longitude']], how='left', on='SN')

        try:
            st.write(f"Generating Excel ...")
            # save to Excel ###################################################
            Excel_io = convert_excel(index_df, data_df, score_df, change_df, tiled_theme, rank=rank)
            with open(f'{output_path}{os.sep}{ofn_header}_Data_Score_Change.xlsx', 'wb') as f:
                f.write(Excel_io.getbuffer())
        except:
            st.write(f"Generating CSVs ...")
            # save to CSVs ###################################################
            convert_csv(index_df, data_df, score_df, change_df, f'{output_path}{os.sep}{ofn_header}', tiled_theme, rank=rank, only_files=True)

        if os.name == 'nt':
            st.write(f"Generating PDFs ...")
            from docx2pdf import convert
            import pythoncom
            pythoncom.CoInitialize()
            try: convert( f'{output_path}{os.sep}' )
            except: pass

        st.write(f"Compressing Reports and Data ...")
        Folder_io = BytesIO()
        with ZipFile(Folder_io, 'w') as archive:
            for fn in glob.glob(f'{output_path}{os.sep}*.*'):
                # if fn.endswith('nc'): continue
                archive.write(fn, arcname=fn.split(os.sep)[-1])

        key = f'Build_{build_time}_{report_type.replace(' ','_')}.zip'
        st.session_state[private_cart_key][key] = Folder_io

        st.download_button(
            label=f'📥 {key}',
            data=Folder_io,
            file_name=key,
            on_click=st.stop,
            key=time.strftime('%Y.%m.%d.%H.%M.%S',time.localtime(time.time())),
            )
        status.update(label="Build completed to download.:point_down:", state="complete", expanded=True)

@st.dialog("Please wait ...")
def indexing(match_time, locate, shp_fd_products, index_fd_products, os_format, private_cart_key, aris=[100, 500], Point=shapely.geometry.Point):

    with st.status(f"Fetching: {len(locate)} locations index..", expanded=True) as status:

        st.write("Seeking Global spatial index ...")
 
        drives_mapping = {
            "/media/Modellingdata/": "Y:\\",
            "/media/Public1/": "Q:\\",
            "/media/Public2/": "M:\\",
            "/media/Public3/": "V:\\",
            "/media/Public4/": "Z:\\",
            "/media/Public5/": "F:\\",
            "/media/Public6/": "L:\\",
            "/media/Public7/": "P:\\",
            "/media/Public8/": "O:\\",
            "/media/Public9/": "N:\\",
            "/":  os.sep,
        }

        region_list = sorted(set(shp_fd_products.ctry_long))

        for i, row in locate.iterrows():
            region = (row['Region'] if 'Region' in locate.columns else row['Country'] if 'Country' in locate.columns else 'global').replace(' ','')
            region = region if region in region_list else 'global'
        
            if all(c in locate for c in ["geoometry"]):
                within = shp_fd_products[shp_fd_products.geometry.intersects(locate.geometry.union_all())]
            elif all(c in locate for c in ["Latitude", "Longitude"]):
                lat, lon = row[['Latitude', 'Longitude']].values.tolist()
                within = shp_fd_products[Point(lon,lat).within(shp_fd_products.geometry)]

            for _, sub_shp in within.iterrows():
                country_domain, ctry_long  = sub_shp[['catchment', 'ctry_long']].values.tolist()
                
                if region.lower()=='global' or ctry_long.lower() == region.lower():
                    for ari in aris:
                        fd = index_fd_products[(index_fd_products['country_domain']==country_domain) & (index_fd_products['ari']==ari)]
                        if len(fd)>0:
                            if ari==100: 
                                col = 'dem_path'
                                if col not in locate.columns: locate[col] = ''
                                path_string = ' '.join( [*([] if str(locate.loc[i, col]) in ['nan', 'None', ''] else [locate.loc[i, col]]), *([str(i) for i in fd['dem_file'].values.tolist() if str(i) not in ['nan', 'None', '']])] )
                                locate.loc[i, 'dem_path'] = Path(path_string).format(drives_mapping) if os_format=='Windows' else path_string
                            
                            col = f'fz{ari}_path'
                            if col not in locate.columns: locate[col] = ''
                            path_string = ' '.join( [*([] if str(locate.loc[i, col]) in ['nan', 'None', ''] else [locate.loc[i, col]]), *([str(i) for i in fd['fzone_file'].values.tolist() if str(i) not in ['nan', 'None', '']])] )
                            locate.loc[i, col] = Path(path_string).format(drives_mapping) if os_format=='Windows' else path_string
                            
                            col = f'fd{ari}_path'
                            if col not in locate.columns: locate[col] = ''
                            path_string = ' '.join( [*([] if str(locate.loc[i, col]) in ['nan', 'None', ''] else [locate.loc[i, col]]), *([str(i) for i in fd['fdepth_file'].values.tolist() if str(i) not in ['nan', 'None', '']])] )
                            locate.loc[i, col] = Path(path_string).format(drives_mapping) if os_format=='Windows' else path_string

        st.write(f"Matching Region storage paths of {os_format} ...")

        CSV_io = BytesIO()
        locate.to_csv(CSV_io, index=False)

        key = f'Match_{match_time}_Flood_Data_Index.csv'
        st.session_state[private_cart_key][key] = CSV_io

        st.download_button(
            label=f'📥 {key}',
            data=CSV_io,
            file_name=key,
            on_click=st.stop,
            key=time.strftime('%Y.%m.%d.%H.%M.%S',time.localtime(time.time())),
            )
        status.update(label="Fetch completed to download.:point_down:", state="complete", expanded=True)

# ----------------------------------------------------------------------------------------
# Non cache resource
# ----------------------------------------------------------------------------------------

def convert_asc(_da, nodata=-9999., make_mask=False):
    if isinstance(nodata, int): digits = 0
    else: digits = 2
    raster_io = BytesIO()
    if make_mask:
        _da = xr.where(np.isnan(_da), 0, 1)
    else:
        _da = xr.where(np.isnan(_da), nodata, _da)
    savetxt(raster_io, *nc2xr(_da, nodata=nodata), digits=digits, silent=True)
    return raster_io

def convert_tif(_da, nodata=-9999., make_mask=False):
    if isinstance(nodata, int): digits = 0
    else: digits = 2
    raster_io = BytesIO()
    if make_mask:
        _da = xr.where(np.isnan(_da), 0, 1)
    else:
        _da = xr.where(np.isnan(_da), nodata, _da)
    savetif(nc2xr(_da, nodata=nodata), raster_io, digits=digits, silent=True)
    return raster_io

def convert_df(_ds, name='Value', nodata=-9999.):
    _ds = _ds.drop_vars(['lat','lon','x','y','spatial_ref'], errors='ignore')
    vars = []
    for vn in _ds.data_vars:
        _da = _ds[vn]
        df = _da.to_dataframe().stack().rename_axis(index={None: 'Indicator'}).rename(name).reset_index()
        if 'cat' in _da.dims: df = df[~df[name].isin([nodata])]
        vars.append(df)
    df = (
        pd.concat(vars)
        .rename(columns={'month':'Month','point':'Address','pth':'Percentile','ssp':'Scenario','year':'Year','ari':'Return period','hrs':'Hour','cat':'Category','depth':'Depth'})
        )
    if 'Address' in df.columns:
        df.insert(0, 'SN', None)
        df_split = df['Address'].astype(str).str.split(pat='<-split->', expand=True)
        if len(df_split.columns)==1:
            df['SN'] = df_split
        elif len(df)>0:
            df[['SN','Address']] = df_split[[0,1]]
    df['Indicator'] = df.pop('Indicator')
    df[name] = df.pop(name)
    return df.replace(np.nan, '-N/A-')

def convert_csv(index_df, data_df, score_df, change_df, ofn_header, tiled_theme, rank=False, only_files=False, ari=100):
    if only_files:
        index_df.replace(['0',''], np.nan).replace(np.nan, '-N/A-').to_csv(f'{ofn_header}_Index.csv', index=False)
        if not tiled_theme:
            theme_df = merge_data_change_score([
                data_df.replace(-9999, np.nan).replace(np.nan, '-N/A-').set_index('SN'),
                score_df.replace(-9999, np.nan).replace(np.nan, '-N/A-').set_index('SN'),
                change_df.replace(-9999, np.nan).replace(np.nan, '-N/A-').set_index('SN'),
                ]).reset_index()
            theme_df.to_csv(f'{ofn_header}_Data_Score_Change.csv', index=False)
        else:
            data_df.replace(-9999, np.nan).replace(np.nan, '-N/A-').to_csv(f'{ofn_header}_Data.csv', index=False)
            score_df.replace(-9999, np.nan).replace(np.nan, '-N/A-').to_csv(f'{ofn_header}_Score.csv', index=False)
            change_df.replace(-9999, np.nan).replace(np.nan, '-N/A-').to_csv(f'{ofn_header}_Change.csv', index=False)

        if rank and len(set(score_df['Address'].values))>1:
            score_df = score_df.replace(-9999, np.nan).replace(np.nan, '-N/A-')
            if "Return period" in score_df.columns: score_df = score_df[score_df['Return period'].isin([ari, str(ari), str(int(ari)), '-N/A-'])]
            for ssp in sorted(set(score_df['Scenario'].values)):
                for pth in sorted(set(score_df['Percentile'].values)):
                    for year in sorted(set(score_df['Year'].values)):
                        tmp = score_df[(score_df['Percentile']==pth)&(score_df['Scenario']==ssp)&(score_df['Year']==year)]
                        tmp['SN'] = tmp['SN'].astype(int)
                        tmp = tmp.pivot(index=['SN','Address'], columns='Indicator', values='Score')
                        tmp['Dense Rank'] = tmp.replace('-N/A-', np.nan).apply(tuple, axis=1).rank(method='dense', ascending=False)
                        tmp = tmp.reset_index()    
                        tmp.to_csv(f'{ofn_header}_Rank_SSP{ssp}_{int(pth)}Pth_{year}.csv', index=False)
    else:        
        CSV_io = BytesIO()
        with ZipFile(CSV_io, 'w') as archive:
            archive.writestr(f'{ofn_header}_Index.csv', index_df.replace(['0',''], np.nan).replace(np.nan, '-N/A-').to_csv(index=False))
            if not tiled_theme:
                theme_df = merge_data_change_score([
                    data_df.replace(-9999, np.nan).replace(np.nan, '-N/A-').set_index('SN'),
                    score_df.replace(-9999, np.nan).replace(np.nan, '-N/A-').set_index('SN'),
                    change_df.replace(-9999, np.nan).replace(np.nan, '-N/A-').set_index('SN'),
                    ]).reset_index()
                archive.writestr(f'{ofn_header}_Data_Score_Change.csv', theme_df.to_csv(index=False))
            else:
                archive.writestr(f'{ofn_header}_Data.csv', data_df.replace(-9999, np.nan).replace(np.nan, '-N/A-').to_csv(index=False))
                archive.writestr(f'{ofn_header}_Score.csv', score_df.replace(-9999, np.nan).replace(np.nan, '-N/A-').to_csv(index=False))
                archive.writestr(f'{ofn_header}_Change.csv', change_df.replace(-9999, np.nan).replace(np.nan, '-N/A-').to_csv(index=False))

            if rank and len(set(score_df['Address'].values))>1:
                score_df = score_df.replace(-9999, np.nan).replace(np.nan, '-N/A-')
                if "Return period" in score_df.columns: score_df = score_df[score_df['Return period'].isin([ari, str(ari), str(int(ari)), '-N/A-'])]
                for ssp in sorted(set(score_df['Scenario'].values)):
                    for pth in sorted(set(score_df['Percentile'].values)):
                        for year in sorted(set(score_df['Year'].values)):
                            tmp = score_df[(score_df['Percentile']==pth)&(score_df['Scenario']==ssp)&(score_df['Year']==year)]
                            tmp['SN'] = tmp['SN'].astype(int)
                            tmp = tmp.pivot(index=['SN','Address'], columns='Indicator', values='Score')
                            tmp['Dense Rank'] = tmp.replace('-N/A-', np.nan).apply(tuple, axis=1).rank(method='dense', ascending=False)
                            tmp = tmp.reset_index()        
                            archive.writestr(f'{ofn_header}_Rank_SSP{ssp}_{int(pth)}Pth_{year}.csv', tmp.to_csv(index=False))
        return CSV_io

def convert_excel(index_df, data_df, score_df, change_df, tiled_theme, rank=False, ari=100):
    Excel_io = BytesIO()
    with pd.ExcelWriter(Excel_io) as writer:

        workbook  = writer.book
        wrap_format = workbook.add_format({'text_wrap': True})
        cover_info_page.to_excel(writer, sheet_name='Info', index=False)
        writer = add_img_excel(writer,  sheet_name='Info', img_filename=f'CRA_Parameter{os.sep}ClimSystems_logo_name_sheet_head.png')
        writer.sheets['Info'].set_column(0, 0, 35)
        writer.sheets['Info'].set_column(1, 1, 125, wrap_format)

        index_df.replace(['0',''], np.nan).replace(np.nan, '-N/A-').to_excel(writer, sheet_name='Climate Index', index=False)
        writer = auto_set_excel(index_df, 'Climate Index', writer)

        if not tiled_theme:
            theme_df = merge_data_change_score([
                data_df.replace(-9999, np.nan).replace(np.nan, '-N/A-').set_index('SN'),
                score_df.replace(-9999, np.nan).replace(np.nan, '-N/A-').set_index('SN'),
                change_df.replace(-9999, np.nan).replace(np.nan, '-N/A-').set_index('SN'),
                ]).reset_index()
            theme_df.to_excel(writer, sheet_name='Data_Score_Change', index=False)
            writer = auto_set_excel(theme_df, 'Data_Score_Change', writer)
        else:
            data_df.replace(-9999, np.nan).replace(np.nan, '-N/A-').to_excel(writer, sheet_name='Data', index=False)
            writer = auto_set_excel(data_df, 'Data', writer)
            change_df.replace(-9999, np.nan).replace(np.nan, '-N/A-').to_excel(writer, sheet_name='Change', index=False)
            writer = auto_set_excel(change_df, 'Change', writer)
            score_df.replace(-9999, np.nan).replace(np.nan, '-N/A-').to_excel(writer, sheet_name='Score', index=False)
            writer = auto_set_excel(score_df, 'Score', writer)

        if rank and len(set(score_df['Address'].values))>1:
            score_df = score_df.replace(-9999, np.nan).replace(np.nan, '-N/A-')
            if "Return period" in score_df.columns: score_df = score_df[score_df['Return period'].isin([ari, str(ari), str(int(ari)), '-N/A-'])].drop(columns='Return period')
            for ssp in sorted(set(score_df['Scenario'].values)):
                for pth in sorted(set(score_df['Percentile'].values)):
                    for year in sorted(set(score_df['Year'].values)):
                        tmp = score_df[(score_df['Percentile']==pth)&(score_df['Scenario']==ssp)&(score_df['Year']==year)]
                        tmp['SN'] = tmp['SN'].astype(int)
                        tmp = tmp.pivot(index=['SN','Address'], columns='Indicator', values='Score')
                        tmp['Dense Rank'] = tmp.replace('-N/A-', np.nan).apply(tuple, axis=1).rank(method='dense', ascending=False)
                        tmp = tmp.reset_index()
                        tmp.to_excel(writer, sheet_name=f'Rank_SSP{ssp}_{int(pth)}Pth_{year}', index=False)
                        writer = auto_set_excel(tmp, f'Rank_SSP{ssp}_{int(pth)}Pth_{year}', writer)
    return Excel_io

def chg_dict_key(d):
    return dict((key.replace("_"," ").replace("~",""), value) for (key, value) in d.items())

def chg_df_index(df):
    df.index = (i.replace("_"," ").replace("~","") for i in df.index)
    return df

def merge_data_change_score(in_obj) -> pd.DataFrame:
    import pandas as pd
    df_dict = {}
    if isinstance(in_obj, str) and (in_obj.endswith('xlsx') or in_obj.endswith('xls')):
        in_obj = pd.read_excel(in_obj, sheet_name=None)    
        for key in in_obj:
            if key in ['Data']: 
                df_dict[key] = in_obj[key]
                df_dict[key].insert(df_dict[key].columns.get_loc('Value'), 'Theme', key)
            elif key in ['Change']: 
                df_dict[key] = in_obj[key].rename(columns={'Change':'Value'})
                df_dict[key].insert(df_dict[key].columns.get_loc('Value'), 'Theme', key)
            elif key in ['Score']: 
                df_dict[key] = in_obj[key].rename(columns={'Score':'Value'})
                df_dict[key].insert(df_dict[key].columns.get_loc('Value'), 'Theme', key)
    elif isinstance(in_obj, (tuple, list)):
        for obj in in_obj:
            df = pd.read_csv(obj, low_memory=False) if isinstance(obj, str) else obj if isinstance(obj, pd.DataFrame) else pd.DataFrame()
            if 'Value' in df.columns: 
                key = 'Data'
                df_dict[key] = df.copy()
                df_dict[key].insert(df_dict[key].columns.get_loc('Value'), 'Theme', key)
            elif 'Change' in df.columns: 
                key = 'Change'
                df_dict[key] = df.copy().rename(columns={key:'Value'})
                df_dict[key].insert(df_dict[key].columns.get_loc('Value'), 'Theme', key)
            elif 'Score' in df.columns: 
                key = 'Score'
                df_dict[key] = df.copy().rename(columns={key:'Value'})
                df_dict[key].insert(df_dict[key].columns.get_loc('Value'), 'Theme', key)
    elif isinstance(in_obj, dict):
        for key in in_obj:
            if key in ['Data', 'Change', 'Score']: 
                df_dict[key] = in_obj[key]
                df_dict[key].insert(df_dict[key].columns.get_loc('Value'), 'Theme', key)
    if df_dict:
        df = pd.concat(df_dict.values())
        if 'Units' not in df.columns:
            name_units = df.Indicator.str.replace(r'[()]', '', regex=True).str.split(' ', expand=True)
            if len(name_units.columns)>1:
                df['Indicator'] = name_units[0]
                df.insert(df.columns.get_loc('Value')+1, 'Units', name_units[1])
        else:
            name_units = df['Units']
            df = df.drop(columns='Units')
            df.insert(df.columns.get_loc('Value')+1, 'Units', name_units)
    else: df = pd.DataFrame()
    return df

# ----------------------------------------------------------------------------------------
# Function module
# ----------------------------------------------------------------------------------------

def Climate_Toolbox(API_KEY='', HOST_v1v2='', HOST_v3='', EMAIL='', LEVEL=0, max_point_limit=20000, main_tab=st.sidebar):

    about = {
        'Distance to Coast': "We provide blazing-fast, highly-accurate, low-cost Distance to Coast calculation for most location of world.",
        'Elevation': "We provide newest, highly-accurate Elevation data for most location of world.",
        'Climate Indicators Data': "We provide the Climate Indicators Data for most location of world.",
        'Climate Indicators Analysis': "We provide the Climate Indicators Analysis for most location of world.",
        'Climate Risk Report': "We provide the Climate Risk Report for most location of world.",
        'Climate Assessment Project': "We provide the Climate Assessment Project for most location of world.",
        'Flood Zone Map': "We provide the Flood Zone Map for most location of world.",
        'Flood Data Index': "Build the Flood Data Index for most location of world.",
        }
    
    params = {
        "key": API_KEY,
        "email": EMAIL,
        "nscore": 10,
        'dist_coast': 5,
        'dist_vegetation': 0.15,
        "format": "dataset",
        "kind": ["data", "change", "score"],
        "year": [2005],
        "hrs":  [24],
        "ari":  [100, 500],
        "pth":  [5, 50, 95],
        "depth":[0],
        "lat":  [],
        "lon":  [],
        "cat":  [],
        "ssp":  [],
        "variable": [],
        "month": list(range(1,13)),
        }

    # ------------------------------------------------------------------------------------
    # Physical Risk Reporting (varialbles list)
    # ------------------------------------------------------------------------------------

    physical_risk_vns = [
        "Cooling_Degree_Days",
        "Extreme_Precipitation",
        "Maximum_Temperature_Days_Higher_35degC",
        # "Rainfall_Flood_Depth", ## remove from 16/10/2024
        "Extreme_Water_Level",
        "Extreme_Wind_Speed",
        "Air_Heatwave_Days",
        "Heating_Degree_Days",
        "KBDI_Fire_Risk",
        "Mean_Sea_Level_Rise",
        "Monthly_Mean_Precipitation",
        "Monthly_Mean_Temperature",
        "Monthly_Relative_Humidity"
        ]
    physical_analysis_vns = [
        'Monthly_Mean_Temperature',
        'Monthly_Maximum_Temperature',
        'Monthly_Minimum_Temperature',
        'Monthly_Sea_Water_Temperature',
        'Monthly_Sea_Surface_Temperature',
        'Cooling_Degree_Days',
        'Maximum_Temperature_Days_Higher_30degC',
        'Maximum_Temperature_Days_Higher_35degC',
        'Maximum_Temperature_Days_Higher_40degC',
        'Warm_Spell_Duration_Index',
        'Air_Heatwave_Days',
        'Marine_Heatwave_Days',
        'Cold_Spell_Duration_Index',
        'Frost_Days',
        'Forest_Fire_Danger_Index',
        'Monthly_Wind_Speed',
        'Extreme_Wind_Speed',
        'Monthly_Mean_Precipitation',
        'Monthly_Mean_Snowfall',
        'Monthly_Mean_Runoff',
        'Monthly_Sea_Water_pH',
        'Mean_Sea_Level_Rise',
        'Extreme_Water_Level',
        'Water_Stress_Category',
        'SPEI_Drought_Probability_3mon',
        'SPEI_Drought_Probability_24mon',
        'Extreme_Precipitation',
        'Landslide'
        ]
    forest_risk_vns = [
        "Monthly_Mean_Temperature",
        "Monthly_Mean_Precipitation",
        "Monthly_Solar_Radiation",
        "Monthly_Relative_Humidity",
        "Monthly_Potential_Evapotranspiration_HG",
        "Monthly_Soil_Temperature",
        "Monthly_Soil_Moisture",
        "Growing_Degree_Days_4degC",
        "Frost_Days",
        "SPEI_Drought_Probability_3mon",
        "Extreme_Precipitation",
        "Extreme_Wind_Speed",
        "Mean_Sea_Level_Rise",
        "Extreme_Water_Level",
        "Air_Heatwave_Days",
        "Maximum_Temperature_Days_Higher_35degC",
        "Cooling_Degree_Days",
        "Heating_Degree_Days",
        "KBDI_Fire_Risk"
        ]
    solar_farm_risk_vns = [
        "Monthly_Maximum_Temperature",
        "Monthly_Mean_Temperature",
        "Monthly_Minimum_Temperature",
        "Monthly_Mean_Precipitation",
        "Monthly_Relative_Humidity",
        "Cooling_Degree_Days",
        "Maximum_Temperature_Days_Higher_25degC",
        "Maximum_Wind_Speed_Hours_Higher_89KM_per_Hour",
        "Air_Heatwave_Days",
        "Frost_Days",
        "Extreme_Precipitation",
        #"Rainfall_Flood_Depth", ## remove from 2/4/2025
        "Extreme_Wind_Speed",
        "KBDI_Fire_Risk",
        "Mean_Sea_Level_Rise",
        "Extreme_Water_Level",
        "SPEI_Drought_Probability_3mon"
        ]
    logos_physical_risk_vns = [
        "Monthly_Maximum_Temperature",
        "Monthly_Minimum_Temperature",
        "Monthly_Mean_Temperature",
        "Monthly_Mean_Precipitation",
        "Monthly_Relative_Humidity",
        "Monthly_Potential_Evapotranspiration_HG",
        "Monthly_Solar_Radiation",
        "Monthly_Soil_Moisture",
        "Air_Heatwave_Days",
        "Extreme_Precipitation",
        "Extreme_Wind_Speed",
        "KBDI_Fire_Risk",
        "Mean_Sea_Level_Rise",
        "Extreme_Water_Level",
        "SPEI_Drought_Probability_3mon"
        ]
    new_zealand_physical_risk_vns = [
        "Cooling_Degree_Days",
        "Extreme_Precipitation",
        "Maximum_Temperature_Days_Higher_25degC",
        # "Rainfall_Flood_Depth", ## remove from 2/4/2025
        "Extreme_Water_Level",
        "Extreme_Wind_Speed",
        "Air_Heatwave_Days",
        "Heating_Degree_Days",
        "KBDI_Fire_Risk",
        "Mean_Sea_Level_Rise",
        "Monthly_Mean_Precipitation",
        "Monthly_Mean_Temperature",
        "Monthly_Relative_Humidity"
        ]
    CSRD_physical_risk_vns = [
        "Air_Heatwave_Days",
        "Cooling_Degree_Days",
        "Extreme_Precipitation",
        "Extreme_Water_Level",
        "Extreme_Wind_Speed",
        "Heating_Degree_Days",
        "KBDI_Fire_Risk",
        "Landslide",
        "Maximum_Temperature_Days_Higher_35degC",
        "Mean_Sea_Level_Rise",
        "Monthly_Mean_Precipitation",
        "Monthly_Mean_Temperature",
        "Monthly_Relative_Humidity",
        #"Rainfall_Flood_Depth", ## remove from 2/4/2025
        'SPEI_Drought_Probability_3mon',
        ]

    # ------------------------------------------------------------------------------------
    # Climate Assessment Project (varialbles list)
    # ------------------------------------------------------------------------------------
    
    alcoa_climate_assessment_vns = [
        "Air_Heatwave_Days",
        'Cooling_Degree_Days',
        'Extreme_Precipitation',
        'Wet_Seasons_Extreme_Precipitation',
        'Extreme_Water_Level',
        'Extreme_Wind_Speed',
        'Heating_Degree_Days',
        'Humidex',
        'Maximum_Temperature_Days_Higher_25degC',
        'Maximum_Temperature_Days_Higher_35degC',
        'Maximum_Temperature_Days_Higher_40degC',
        'Maximum_Temperature_Days_Higher_45degC',
        'Mean_Sea_Level_Rise',
        'Monthly_Maximum_Temperature',
        'Monthly_Mean_Precipitation',
        'Monthly_Mean_Temperature',
        'Monthly_Minimum_Temperature',
        'Monthly_Wind_Speed',
        'Probable_Maximum_Precipitation',
        'SPEI_Drought_Probability_12mon',
        'SPEI_Drought_Probability_1mon',
        'SPEI_Drought_Probability_24mon',
        'SPEI_Drought_Probability_3mon',
        'SPEI_Drought_Probability_6mon',
        ]
    colonial_climate_assessment_vns = [
        'Extreme_Precipitation',
        'Extreme_Water_Level',
        'Extreme_Wind_Speed',
        'Fire_Weather_Index',
        'Frost_Days',
        'Landslide',
        'Maximum_Temperature_Days_Higher_25degC',
        'Mean_Sea_Level_Rise',
        'SPEI_Drought_Probability_3mon',
        'Monthly_Mean_Snow_Depth',
        ]

    # ------------------------------------------------------------------------------------
    # Climate Indicators Analysis (varialbles list)
    # ------------------------------------------------------------------------------------
    
    climate_indicators_analysis_vns = [
        'Air_Heatwave_Days', 'Cold_Spell_Duration_Index',
       'Cooling_Degree_Days', 'Extreme_Precipitation',
       'Extreme_Water_Level', 'Extreme_Wind_Speed',
       'Forest_Fire_Danger_Index', 'Frost_Days', 'Heating_Degree_Days',
       'KBDI_Fire_Risk', 'Landslide', 'Marine_Heatwave_Days',
       'Maximum_Temperature_Days_Higher_30degC',
       'Maximum_Temperature_Days_Higher_35degC',
       'Maximum_Temperature_Days_Higher_40degC', 'Mean_Sea_Level_Rise',
       'Monthly_Maximum_Temperature', 'Monthly_Mean_Precipitation',
       'Monthly_Mean_Runoff', 'Monthly_Mean_Snowfall',
       'Monthly_Mean_Temperature', 'Monthly_Minimum_Temperature',
       'Monthly_Relative_Humidity', 'Monthly_Sea_Surface_Temperature',
       'Monthly_Sea_Water_Temperature', 'Monthly_Sea_Water_pH',
       'Monthly_Wind_Speed', 'SPEI_Drought_Probability_24mon',
       'SPEI_Drought_Probability_3mon', 'Warm_Spell_Duration_Index',
       'Water_Stress_Category']

    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
 
    private_cart_key = f'Private_Cart_{EMAIL}'

    # initialization st.session_state
    if private_cart_key not in st.session_state:
        st.session_state[private_cart_key] = {}
    if 'countries_shp' not in st.session_state:
        st.session_state['countries_shp'] = load_shape_data(f'CRA_Parameter{os.sep}countries.dbf')
    if 'VaR_calculate_parameter' not in st.session_state:
        st.session_state['VaR_calculate_parameter'] = load_excel(f'CRA_report{os.sep}VaR_calculate_parameters.xlsx', sheet_name=None, index_col=0)
    if 'index_fd_products' not in st.session_state:
        st.session_state['index_fd_products'] = load_csv(f'CRA_Parameter{os.sep}Flood_Products_log.csv')
    if 'shp_fd_products' not in st.session_state:
        st.session_state['shp_fd_products'] = load_shape_data(f'CRA_Parameter{os.sep}Flood_Spatial_index.dbf')

    if (st.session_state['countries_shp'] is None or
        st.session_state['VaR_calculate_parameter'] is None or
        st.session_state['index_fd_products'] is None or
        st.session_state['shp_fd_products'] is None
        ): st.error('⚠️ Server initialization error, please contact administrator.')

    countries_shp, VaR_calculate_parameters, index_fd_products, shp_fd_products = st.session_state['countries_shp'], st.session_state['VaR_calculate_parameter'], st.session_state['index_fd_products'], st.session_state['shp_fd_products']

    month_numb = [f'{i:02d}' for i in range(1,13)]
    month_abbr = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    locate = pd.DataFrame([(0.,0.)], columns=['Latitude','Longitude'])

    build_able = True
    address = None
    within = True
    geo = None
    r = None

    flood_map = None
    flood_scheme_path = None

    user_access_level = {
        0: ['Climate Indicators Data', 'Climate Indicators Analysis', 'Climate Risk Report', 'Climate Assessment Project', 'Flood Data Index', 'Flood Zone Map', 'Distance to Coast', 'Elevation'],
        1: ['Climate Indicators Data', 'Climate Indicators Analysis', 'Climate Risk Report', 'Climate Assessment Project', 'Flood Data Index'],
        2: ['Climate Indicators Data', 'Climate Indicators Analysis', 'Flood Data Index'],
        3: ['Climate Indicators Data', 'Climate Indicators Analysis', 'Flood Data Index'],
        4: ['Flood Data Index', 'Distance to Coast', 'Elevation'],
        }

    header = main_tab.selectbox('Select a tool:', user_access_level[int(LEVEL)])
    st.header(header)
    st.caption("👈 Welcome to the World's Leading Climate and Environmental Data Services System.")
    map_header_cols = st.columns(5)
    build_private_cart(private_cart_key, map_header_cols[4])
    type_select = main_tab.empty()

    with map_header_cols[0].popover(':gear: Settings', use_container_width=False):

        if header == 'Elevation':
            version = {'NASA_SRTM_v3.0_Global_30m_DEM':'v1', 'ALOS_AW3D30_v3.2_Global_30m_DEM':'v3'}[ st.selectbox('Project:', ['ALOS_AW3D30_v3.2_Global_30m_DEM','NASA_SRTM_v3.0_Global_30m_DEM'], 0, help='Select a project.') ]

        elif header == 'Flood Zone Map':
            flood_scheme_path = st.selectbox('Source data:', sorted(glob.glob(f'Flood_data{os.sep}*')), 0, help='Select a scheme source data.')
            with st.form("scheme_form"):
                flood_map_type = st.selectbox('Map type:', ["Street", "Satellite", 'terrain'], 0, help='Select a map type.')
                flood_map_size = st.selectbox('Map size:', ['1.5*1 km','3*2 km','5*3 km','10*6 km','50*30 km'], 2, help='Select a size (width * height).')
                flood_map_aris = sorted([int(re.findall(r'\d+', fn.split(os.sep)[-1])[0]) for fn in glob.glob(f'{flood_scheme_path}{os.sep}*.csv')])
                flood_map_aris = sorted(st.multiselect('Return periods (Year):', flood_map_aris, flood_map_aris[:4], max_selections=4, help='Select 1~4 flood return periods.'))

                st.form_submit_button("Submit")

            st.write('Colors:')

            # flood_map_colors = ['#06A6F9', '#57F3FB', '#40E0D0','#B3F1CB'] # defined myself
            # flood_map_colors = ['#5BBCF3', '#EAD359', '#F2F594', '#F5898B'] # FEMA Flood Insurance Floodplain Zones
            # flood_map_colors = ['#B0F999', '#F5898B', '#F2F594', '#E379F3'] # BBNEP Study of FloodPlain Expansion
            flood_map_colors = ['#5BBCF3', '#8CF7F4', '#EAD359', '#E5B3F1'] # SLOSH Model Hurricane Categories

            cols = st.columns(len(flood_map_aris) if flood_map_aris else len(flood_map_colors))
            flood_map_cmap = ListedColormap([
                 cols[i].color_picker(f'{ari}:', flood_map_colors[i%4], help=f'Select a color for ARI {ari} flood layer.') for i, ari in enumerate(flood_map_aris)
                 ])

        elif header == 'Flood Data Index':
            os_format = st.selectbox('Path format:', ['Windows','Linux'], 0, help='Select the path format on operating system.')

        elif header == 'Climate Indicators Data':
            if LEVEL in [0,1]: version = {'CMIP5 - v1':'v1', 'CMIP6 - v2':'v2', 'CMIP6 - v3':'v3'}[ st.selectbox('Project:', ['CMIP6 - v3','CMIP6 - v2','CMIP5 - v1'], 0, help='Select a project.') ]
            else: version = {'CMIP5 - v1':'v1', 'CMIP6 - v2':'v2', 'CMIP6 - v3':'v3'}[ st.selectbox('Project:', ['CMIP6 - v3'], 0, help='Select a project.') ]

            HOST_v3 = api_urls[ st.selectbox('API:', api_urls.keys(), 0, help='Select a API URL address.') ] if version in ['v3'] else HOST_v3

            with st.form("options_form"):
                if version in ['v3']:

                    r_global = requests.post(f'{HOST_v3}/api/v3/global/', json=params).json()

                    tiled_theme = st.checkbox('Tiled Theme (Multi-sheets: Data, Change, Score)', value=False, help='By checked, theme is splitted 3 sheets, otherwise its are in one sheet.')
                    rank = st.checkbox('Calculate risk ranks', help='Calculate dense rank of risk scores, the upload file must contains "Address" column.')
                    nscore = st.selectbox('Risk levels:', [5,10], 1, help='Select a risk level.')

                    dist_coast = st.number_input('Effective distance to coast (km):', value=5., help='Set the distance to coast for some indicator analysis, as Mean Sea Level Rise and Extreme Water Level.')
                    dist_vegetation = st.number_input('Effective distance to vegetation (km):', value=0.15, help='Set the distance to vegetation for some indicator analysis, as Fire Danger Index.')

                    index_list = st.multiselect('Climate Index:', api_index[HOST_v3], api_index[HOST_v3], help='Select multiple climate indexes or none.')
                    scenarios = st.multiselect('Scenarios:', [f'SSP {str(s)[0]}-{str(s)[1]}.{str(s)[2]}' for s in sorted(set([119] + r_global['ssp']))], [f'SSP {str(s)[0]}-{str(s)[1]}.{str(s)[2]}' for s in [126,245,585]], help='Select 2 or more scenarios.')
                    pths = st.multiselect('Pecentages:', r_global['pth'], [5,50,95], help='Select 3 or more pecentages must contains "5,50,95".')
                    aris = st.multiselect('Return periods (Year):', r_global['ari'], [100], help='Select the average recurrence intervals.')
                    hrss = st.multiselect('Frequencys (Hour):', r_global['hrs'], [24], help='Select the frequencys.')
                    depths = st.multiselect('Depths (cm):', r_global['depth'], [r_global['depth'][0]], help='Select the depths.')
                    months = st.multiselect('Months:', r_global['month'], r_global['month'], help='Select the months.')

                    years_multiselect_container = st.container()
                    years = sorted(
                        years_multiselect_container.multiselect('Years:', sorted(set(r_global['year'] + [2100])), sorted(set(r_global['year'] + [2100])), help='Select 2 or more years, the first year will be defined the Baseline year.') if st.checkbox(f"All years ({len(set(r_global['year'] + [2100]))})") else
                        years_multiselect_container.multiselect('Years:', sorted(set(r_global['year'] + [2100])), [2005,2030,2050,2070], help='Select 2 or more years, the first year will be defined the Baseline year.')
                    )
                    variables_multiselect_container = st.container()
                    variables = sorted(
                        variables_multiselect_container.multiselect('Variables:', sorted(r_global['variable']), sorted(r_global['variable']), help='Select 2 or more variables.') if st.checkbox(f"All variables ({len(r_global['variable'])})") else
                        variables_multiselect_container.multiselect('Variables:', sorted(r_global['variable']), [i for i in r_global['variable'] if i in physical_risk_vns], help='Select 2 or more variables.')
                    )

                    params.update(
                        {
                        'ssp': [int(re.sub(r'\D+', '', scenario)) for scenario in scenarios],
                        'pth': pths,
                        'nscore': nscore,
                        'dist_coast': dist_coast,
                        'dist_vegetation': dist_vegetation,
                        'ari': aris,
                        'hrs': hrss,
                        'depth': depths,
                        'month': months,
                        'year': years,
                        'variable': variables,
                        }
                    )

                else: # v1 or v2
                    nscores = [st.selectbox('Risk levels:', [5,10], 1, help='Select a risk levels.')]

                    scenarios = st.multiselect('Scenarios:', {'v1':['RCP 2.6','RCP 4.5','RCP 8.5'], 'v2':['SSP 1-1.9','SSP 1-2.6','SSP 2-4.5','SSP 3-7.0','SSP 5-8.5']}[ version ], {'v1':['RCP 4.5'], 'v2':['SSP 2-4.5']}[ version ], help='Select the scenarios.')
                    scenarios = [scenario.upper().replace('.','').replace('-','').replace(' ','=') if version=='v1' else scenario.lower().replace('.','').replace('-','').replace(' ','=') for scenario in scenarios]

                    pths = st.multiselect('Pecentages:', [5,50,95], [50], help='Select the pecentages.')

                    r_global = requests.get(f"{HOST_v1v2}/api/{version}/global/?key={YOUR_API_KEY}&cat=__no__").json()
                    vn_global = [i for i in r_global['variable'] if 'Cat' not in i]
                    vn_pcrr = requests.get(f"{HOST_v1v2}/api/{version}/pcrr/?key={YOUR_API_KEY}&cat=__no__").json()['variable']

                    aris = st.multiselect('Return periods (Year):', r_global['ARI'] if version=='v1' else r_global['ari'], [100], help='Select the average recurrence intervals.')

                    hrss = st.multiselect('Frequencys (Hour):', r_global['Hrs'] if version=='v1' else r_global['hrs'], [24], help='Select the frequencys.')

                    depths = st.multiselect('Depths (cm):', r_global['depth'], [r_global['depth'][0]], help='Select the depths.')

                    years = range(1995, 2101, 5) if version=='v1' else range(2005, 2101, 5)
                    years = '+'.join([str(i) for i in st.multiselect('Years:', years, [years[0]] + list(range(2020, 2101, 10)), help='Select the years, the first year will be defined the Baseline year.')])
                    years = f'Year={years}' if version=='v1' else f'year={years}'

                    variables = '+'.join([i for i in sorted(st.multiselect('Variables', vn_global, vn_pcrr, help='Select the variables.'))])
                    variables = f'variable={variables}'

                st.form_submit_button("Submit")

        elif header == 'Climate Indicators Analysis':
            report_type = type_select.selectbox('Type:', [i.split(os.sep)[-1].replace('.xlsx','').replace('_',' ') for i in sorted(glob.glob(f'CRA_report{os.sep}template_parameters{os.sep}*Analysis*.xlsx'))], 0, help='Select a report type.')

            version = {'CMIP6 - v3':'v3'}[ st.selectbox('Project:', ['CMIP6 - v3'], 0, help='Select a project.') ]

            HOST_v3 = api_urls[ st.selectbox('API:', api_urls.keys(), 0, help='Select a API URL address.') ] if version in ['v3'] else HOST_v3
            # flood_scheme_path = st.selectbox('Source data:', sorted(glob.glob(f'Flood_data{os.sep}*')), 0, help='Select a scheme source data.')

            r_global = requests.post(f'{HOST_v3}/api/v3/global/', json=params).json()

            tiled_theme = st.checkbox('Tiled Theme (Multi-sheets: Data, Change, Score)', value=False, help='By checked, theme is splitted 3 sheets, otherwise its are in one sheet.')
            rank = st.checkbox('Calculate risk ranks', help='Calculate dense rank of risk scores, the upload file must contains "Address" column.')
            scenarios_multiselect_container = st.container()
            scenarios = sorted(
                scenarios_multiselect_container.multiselect('Scenarios:', [f'SSP {str(s)[0]}-{str(s)[1]}.{str(s)[2]}' for s in sorted(set([119] + r_global['ssp']))], [f'SSP {str(s)[0]}-{str(s)[1]}.{str(s)[2]}' for s in sorted(set([119] + r_global['ssp']))], help='Select 2 or more scenarios.') if st.checkbox(f"All scenarios ({len(set([119] + r_global['ssp']))})") else
                scenarios_multiselect_container.multiselect('Scenarios:', [f'SSP {str(s)[0]}-{str(s)[1]}.{str(s)[2]}' for s in sorted(set([119] + r_global['ssp']))], [f'SSP {str(s)[0]}-{str(s)[1]}.{str(s)[2]}' for s in [126,245,585]], max_selections=5, help='Select 2 or more scenarios.')
            )
            years_multiselect_container = st.container()
            years = sorted(
                years_multiselect_container.multiselect('Years:', sorted(set(r_global['year'] + [2100])), sorted(set(r_global['year'] + [2100])), help='Select 2 or more years, the first year will be defined the Baseline year.') if st.checkbox(f"All years ({len(set(r_global['year'] + [2100]))})") else
                years_multiselect_container.multiselect('Years:', sorted(set(r_global['year'] + [2100])), [2005,2030,2050,2070,2100], help='Select 2 or more years, the first year will be defined the Baseline year.')
            )

            if report_type in ['General Climate Indicators Analysis']:

                variables_multiselect_container = st.container()
                variables = sorted(
                    variables_multiselect_container.multiselect('Variables:', sorted(r_global['variable']), sorted(r_global['variable']), help='Select 2 or more variables.') if st.checkbox(f"All variables ({len(r_global['variable'])})") else
                    variables_multiselect_container.multiselect('Variables:', sorted(r_global['variable']), [i for i in r_global['variable'] if i in climate_indicators_analysis_vns], help='Select 2 or more variables.')
                )

                params.update(
                    {
                    'ssp': [int(re.sub(r'\D+', '', scenario)) for scenario in scenarios],
                    'ari': [100, 500],
                    'year': years,
                    'variable': variables,
                    }
                )

            elif report_type in ['CSRD Climate Indicators Analysis']:

                params.update(
                    {
                    'ssp': [int(re.sub(r'\D+', '', scenario)) for scenario in scenarios],
                    'ari': [100, 500],
                    'year': years,
                    'variable': climate_indicators_analysis_vns,
                    }
                )

            years = f"year={'+'.join([str(year) for year in years])}"

        elif header == 'Climate Risk Report':
            report_type = type_select.selectbox('Type:', [i.split(os.sep)[-1].replace('.xlsx','').replace('_',' ') for i in sorted(glob.glob(f'CRA_report{os.sep}template_parameters{os.sep}*Risk*.xlsx'))], 0, help='Select a report type.')

            if LEVEL in [0,1]: version = {'CMIP5 - v1':'v1', 'CMIP6 - v2':'v2', 'CMIP6 - v3':'v3'}[ st.selectbox('Project:', ['CMIP6 - v3', 'CMIP6 - v2'], 0, help='Select a project.') ]
            else: version = {'CMIP5 - v1':'v1', 'CMIP6 - v2':'v2', 'CMIP6 - v3':'v3'}[ st.selectbox('Project:', ['CMIP6 - v3'], 0, help='Select a project.') ]

            HOST_v3 = api_urls[ st.selectbox('API:', api_urls.keys(), 0, help='Select a API URL address.') ] if version in ['v3'] else HOST_v3

            r_global = requests.post(f'{HOST_v3}/api/v3/global/', json=params).json()

            tiled_theme = st.checkbox('Tiled Theme (Multi-sheets: Data, Change, Score)', value=False, help='By checked, theme is splitted 3 sheets, otherwise its are in one sheet.')
            rank = st.checkbox('Calculate risk ranks', help='Calculate dense rank of risk scores, the upload file must contains "Address" column.')
            scenarios = st.multiselect('Scenarios:', [f'SSP {str(s)[0]}-{str(s)[1]}.{str(s)[2]}' for s in sorted(set([119] + r_global['ssp']))], [f'SSP {str(s)[0]}-{str(s)[1]}.{str(s)[2]}' for s in [126,245,585]], max_selections=5, help='Select 2 or more scenarios.')
            years = st.multiselect('Years:', sorted(set(r_global['year'] + [2100])), [2005,2030,2050,2070], max_selections=5, help='Select 2 or more years, the first year will be defined the Baseline year.')

            if   report_type in ['Physical Risk Report', 'Physical Risk VaR Report', 'Hotel and Motel Physical Risk Report']:
                
                params.update(
                    {
                    'ssp': [int(re.sub(r'\D+', '', scenario)) for scenario in scenarios],
                    'ari': [100],
                    'year': years,
                    'variable': physical_risk_vns + ["Rainfall_Flood_Depth"] if report_type in ['Physical Risk VaR Report'] else physical_risk_vns,
                    }
                )

                scenarios = [f"ssp={i}" for i in scenarios]
                pths = [5,50,95]
                nscores = [10]
                aris = [100]
                hrss = [24]
                depths = [7]
                cat = ''
                years = f"year={'+'.join([str(year) for year in years])}"
                variables = '+'.join(physical_risk_vns)
                variables = f'variable={variables}'

            elif report_type in ['Physical Risk with Maps Report']:

                flood_scheme_path = st.selectbox('Source data:', sorted(glob.glob(f'Flood_data{os.sep}*')), 0, help='Select a scheme source data.')

                params.update(
                    {
                    'ssp': [int(re.sub(r'\D+', '', scenario)) for scenario in scenarios],
                    'ari': [100],
                    'year': years,
                    'variable': physical_risk_vns + ["Rainfall_Flood_Depth"],
                    }
                )

                scenarios = [f"ssp={i}" for i in scenarios]
                pths = [5,50,95]
                nscores = [10]
                aris = [100]
                hrss = [24]
                depths = [7]
                cat = ''
                years = f"year={'+'.join([str(year) for year in years])}"
                variables = '+'.join(physical_risk_vns)
                variables = f'variable={variables}'

            elif report_type in ['Logos Physical Risk with Maps Report']:
                flood_scheme_path = st.selectbox('Source data:', sorted(glob.glob(f'Flood_data{os.sep}*')), 0, help='Select a scheme source data.')

                params.update(
                    {
                    'ssp': [int(re.sub(r'\D+', '', scenario)) for scenario in scenarios],
                    'cat': ["Extreme_Dry","Severe_Dry","Moderate_Dry"],
                    'ari': [100],
                    'year': years,
                    'variable': logos_physical_risk_vns,
                    }
                )

                scenarios = [f"ssp={i}" for i in scenarios]
                pths = [5,50,95]
                nscores = [10]
                aris = [100]
                hrss = [24]
                depths = [7]
                cat = "cat=Extreme_Dry+Severe_Dry+Moderate_Dry"
                years = f"year={'+'.join([str(year) for year in years])}"
                variables = '+'.join(logos_physical_risk_vns)
                variables = f'variable={variables}'

            elif report_type in ['New Zealand Physical Risk Report', 'New Zealand Physical Risk with FloodZone Report']:
                flood_scheme_path = st.selectbox('Source data:', sorted(glob.glob(f'Flood_data{os.sep}*')), 0, help='Select a scheme source data.')

                params.update(
                    {
                    'ssp': [int(re.sub(r'\D+', '', scenario)) for scenario in scenarios],
                    'ari': [100],
                    'year': years,
                    'variable': new_zealand_physical_risk_vns + ["Rainfall_Flood_Depth"] if report_type in ['New Zealand Physical Risk with FloodZone Report'] else new_zealand_physical_risk_vns,
                    }
                )

                scenarios = [f"ssp={i}" for i in scenarios]
                pths = [5,50,95]
                nscores = [10]
                aris = [100]
                hrss = [24]
                depths = [7]
                cat = ''
                years = f"year={'+'.join([str(year) for year in years])}"
                variables = '+'.join(physical_risk_vns)
                variables = f'variable={variables}'

            elif report_type in ['Forest Risk Report']:

                params.update(
                    {
                    'ssp': [int(re.sub(r'\D+', '', scenario)) for scenario in scenarios],
                    'ari': [100],
                    'year': years,
                    'variable': forest_risk_vns,
                    "cat": ["Extreme_Dry","Severe_Dry","Moderate_Dry"],
                    }
                )

                scenarios = [f"ssp={i}" for i in scenarios]
                pths = [5,50,95]
                nscores = [10]
                aris = [100]
                hrss = [24]
                depths = [7]
                cat = "cat=Extreme_Dry+Severe_Dry+Moderate_Dry"
                years = f"year={'+'.join([str(year) for year in years])}"
                variables = '+'.join(forest_risk_vns)
                variables = f'variable={variables}'

            elif report_type in ['Solar Farm Risk Report']:

                flood_scheme_path = st.selectbox('Source data:', sorted(glob.glob(f'Flood_data{os.sep}*')), 0, help='Select a scheme source data.')

                params.update(
                    {
                    'ssp': [int(re.sub(r'\D+', '', scenario)) for scenario in scenarios],
                    'ari': [100],
                    'year': years,
                    "variable": solar_farm_risk_vns,
                    "cat": ["Extreme_Dry","Severe_Dry","Moderate_Dry"],
                    }
                )

                scenarios = [f"ssp={i}" for i in scenarios]
                pths = [5,50,95]
                nscores = [10]
                aris = [100]
                hrss = [24]
                depths = [7]
                cat = "cat=Extreme_Dry+Severe_Dry+Moderate_Dry"
                years = f"year={'+'.join([str(year) for year in years])}"
                variables = '+'.join(solar_farm_risk_vns)
                variables = f'variable={variables}'

            elif report_type in ['CSRD Physical Risk Report', 'CSRD Physical Risk VaR Report']:

                params.update(
                    {
                    'ssp': [int(re.sub(r'\D+', '', scenario)) for scenario in scenarios],
                    'ari': [100],
                    'year': years,
                    "variable": CSRD_physical_risk_vns,
                    "cat": ["Extreme_Dry","Severe_Dry","Moderate_Dry"],
                    }
                )

                scenarios = [f"ssp={i}" for i in scenarios]
                pths = [5,50,95]
                nscores = [10]
                aris = [100]
                hrss = [24]
                depths = [7]
                cat = "cat=Extreme_Dry+Severe_Dry+Moderate_Dry"
                years = f"year={'+'.join([str(year) for year in years])}"
                variables = '+'.join(CSRD_physical_risk_vns)
                variables = f'variable={variables}'

        elif header == 'Climate Assessment Project':
            report_type = type_select.selectbox('Type:', [i.split(os.sep)[-1].replace('.xlsx','').replace('_',' ') for i in sorted(glob.glob(f'CRA_report{os.sep}template_parameters{os.sep}*Assessment*.xlsx'))], 0, help='Select a report type.')

            version = {'CMIP6 - v3':'v3'}[ st.selectbox('Project:', ['CMIP6 - v3'], 0, help='Select a project.') ]

            HOST_v3 = api_urls[ st.selectbox('API:', api_urls.keys(), 0, help='Select a API URL address.') ] if version in ['v3'] else HOST_v3
            # flood_scheme_path = st.selectbox('Source data:', sorted(glob.glob(f'Flood_data{os.sep}*')), 0, help='Select a scheme source data.')

            r_global = requests.post(f'{HOST_v3}/api/v3/global/', json=params).json()

            tiled_theme = st.checkbox('Tiled Theme (Multi-sheets: Data, Change, Score)', value=False, help='By checked, theme is splitted 3 sheets, otherwise its are in one sheet.')
            rank = st.checkbox('Calculate risk ranks', help='Calculate dense rank of risk scores, the upload file must contains "Address" column.')
            scenarios_multiselect_container = st.container()
            scenarios = sorted(
                scenarios_multiselect_container.multiselect('Scenarios:', [f'SSP {str(s)[0]}-{str(s)[1]}.{str(s)[2]}' for s in sorted(set([119] + r_global['ssp']))], [f'SSP {str(s)[0]}-{str(s)[1]}.{str(s)[2]}' for s in sorted(set([119] + r_global['ssp']))], help='Select 2 or more scenarios.') if st.checkbox(f"All scenarios ({len(set([119] + r_global['ssp']))})") else
                scenarios_multiselect_container.multiselect('Scenarios:', [f'SSP {str(s)[0]}-{str(s)[1]}.{str(s)[2]}' for s in sorted(set([119] + r_global['ssp']))], [f'SSP {str(s)[0]}-{str(s)[1]}.{str(s)[2]}' for s in [126,245,585]], max_selections=5, help='Select 2 or more scenarios.')
            )
            years_multiselect_container = st.container()
            years = sorted(
                years_multiselect_container.multiselect('Years:', sorted(set(r_global['year'] + [2100])), sorted(set(r_global['year'] + [2100])), help='Select 2 or more years, the first year will be defined the Baseline year.') if st.checkbox(f"All years ({len(set(r_global['year'] + [2100]))})") else
                years_multiselect_container.multiselect('Years:', sorted(set(r_global['year'] + [2100])), [2005,2030,2050,2070,2100], help='Select 2 or more years, the first year will be defined the Baseline year.')
            )

            if report_type in ['Alcoa Climate Assessment Project']:

                params.update(
                    {
                    'ssp': [int(re.sub(r'\D+', '', scenario)) for scenario in scenarios],
                    'ari': [100],
                    'year': years,
                    "variable": alcoa_climate_assessment_vns,
                    "hrs": [3,6,12,24,48,72,120,144,168],
                    "ari": [2,3,5,10,20,25,50,100,200,300,500,1000,2500,5000,10000],
                    }
                )

            if report_type in ['Colonial Climate Assessment Project']:

                params.update(
                    {
                    'ssp': [int(re.sub(r'\D+', '', scenario)) for scenario in scenarios],
                    'ari': [100],
                    'year': years,
                    'variable': colonial_climate_assessment_vns,
                    }
                )

            years = f"year={'+'.join([str(year) for year in years])}"

    with main_tab.expander('Search for locations:', expanded=True):

        if header in ['Flood Data Index']:
            page = st.radio('By:', ['address','coordinates','file • CSV', 'file • Shapefile'], index=0, help='Select "address" and "coordinates" will get 5 Google recommended locations, select "file • CSV" will get precise positions, select "file • Shapefile" will get precise polygons.')
        elif header in ['Climate Indicators Data', 'Climate Indicators Analysis', 'Climate Risk Report']:
            page = st.radio('By:', ['address','coordinates','file • CSV'], index=0, help='Select "address" and "coordinates" will get 5 Google recommended locations, select "file • CSV" will get precise positions.')
        elif header in ['Climate Assessment Project']:
            page = st.radio('By:', ['file • CSV'], index=0, help='Select "file • CSV" will get precise positions.')
        else:
            page = st.radio('By:', ['address','coordinates'], index=0, help='Select "address" and "coordinates" will get 5 Google recommended locations, may be not precise positions.')

        uploaded_file = None
        with st.form("search_form"):

            if page == 'coordinates':
                lat = st.number_input('Latitude:', min_value=-90., max_value=90., format='%.6f', value=locate['Latitude'].values[0], help='Enter your qurey point latitude.')
                lng = st.number_input('Longitude:', min_value=-180., max_value=180., format='%.6f', value=locate['Longitude'].values[0], help='Enter your qurey point longitude.')
                locate = pd.DataFrame([(lat,lng)],columns=['Latitude','Longitude'])
                address = get_address(lat,lng)
                address = address[0] if address else (lat,lng)
                geo = True

            elif page == 'address':
                address = st.text_input(
                    'Enter address keywords:',
                    help='Enter several address keywords, or Google recommended address.'
                    )

            elif page == 'file • CSV':
                uploaded_file = st.file_uploader('📤', type=['csv'], accept_multiple_files=False, help='The upload file must contains "Latitude" and "Longitude" columns; if contains "Address" column that will be apply.')

            # elif page == 'file • Shapefile':
            #     uploaded_file = st.file_uploader('📤', type=['csv'], accept_multiple_files=False, help='The upload file must contains "Latitude" and "Longitude" columns; if contains "Address" column that will be apply.')

            search_botton = st.form_submit_button("Search")

            if uploaded_file is not None:
                info = pd.read_csv(uploaded_file, encoding_errors='replace').fillna(0)
                if len(info.index) > max_point_limit:
                    st.error(f'The locations can not more than {max_point_limit} for this version.')
                    st.stop()
                st.dataframe(info)

                if set(['Latitude', 'Longitude']) <= set(info.columns):
                    locate = info
                else:
                    st.warning(f'⚠️ The file • CSV must contains "Latitude, Longitude" columns.')
                    build_able = False

                if header in ['Climate Indicators Analysis', 'Climate Risk Report', 'Climate Assessment Project']:
                    if report_type in ['Physical Risk VaR Report', 'CSRD Physical Risk VaR Report']:
                        if not set(['Latitude', 'Longitude', 'Address', 'Country', 'Build type', 'Function type', 'Report type']) <= set(info.columns):
                            st.warning(f'⚠️ The file • CSV must containes "Latitude, Longitude, Address, Country, Build type, Function type, Report type" columns for {report_type}.')
                            build_able = False
                    else:
                        if not set(['Latitude', 'Longitude', 'Address', 'Report type']) <= set(info.columns):
                            st.warning(f'⚠️ The file • CSV must containes "Latitude, Longitude, Address, Report type" columns for {report_type}.')
                            build_able = False

                if 'Address' in info.columns:
                    info['Address'] = [str(i).translate(translationTable).strip() for i in info['Address']]

        if not isinstance(address, (tuple)) and address:
            geo = get_locations(address)
            if geo is not None:
                address = st.selectbox(
                    'Change address:',
                    geo,
                    help='Change other address in list.'
                    )
                geo = get_coords(address)
                if geo is not None and page == 'address':
                    locate = pd.DataFrame([geo],columns=['Latitude','Longitude'])
            if geo is None: st.warning(f'⚠️ The provided "{address}" is not contained within the {header} database.')

            if header == 'Climate Risk Report':
                if report_type in ['Physical Risk VaR Report', 'CSRD Physical Risk VaR Report']:
                    locate['Country'] = st.selectbox('Country:', sorted(VaR_calculate_parameters['Cap'].index), sorted(VaR_calculate_parameters['Cap'].index).index(address.split(',')[-1].strip()) if address.split(',')[-1].strip() in sorted(VaR_calculate_parameters['Cap'].index) else 0, help='Select a country.')
                    locate['Build type'] = st.selectbox('Build type:', sorted(VaR_calculate_parameters['EAD'].index), 0, help='Select a build type.')
                    locate['Report type'] = locate['Function type'] = st.selectbox('Function type:', sorted(VaR_calculate_parameters['PL'].index), 0, help='Select a function type.')
                elif report_type in ['Forest Risk Report']:
                    locate['Report type'] = st.selectbox('Template:', ['Forest'], 0, help='Select a report type.')
                elif report_type in ['Solar Farm Risk Report']:
                    locate['Report type'] = st.selectbox('Template:', ['Solar Farm'], 0, help='Select a report type.')
                elif report_type in ['Logos Physical Risk with Maps Report']:
                    locate['Report type'] = st.selectbox('Template:', ['Logos'], 0, help='Select a report type.')
                elif report_type in ['Hotel and Motel Physical Risk Report']:
                    locate['Report type'] = st.selectbox('Template:', ['Hotel and Motel'], 0, help='Select a report type.')
                else:
                    locate['Report type'] = st.selectbox('Report type:', sorted(VaR_calculate_parameters['PL'].index), 0, help='Select a report type.')
            elif header == 'Climate Assessment Project':
                if report_type in ['Alcoa Climate Assessment Project', 'Colonial Climate Assessment Project']:
                    locate['Report type'] = st.selectbox('Template:', ['Climate Changes'], 0, help='Select a report type.')
            elif header == 'Climate Indicators Analysis':
                if report_type in ['General Climate Indicators Analysis', 'CSRD Climate Indicators Analysis']:
                    locate['Report type'] = st.selectbox('Template:', ['Climate Changes'], 0, help='Select a report type.')

            locate['Address'] = address

        elif uploaded_file is not None:
            geo = True

        else: geo = False

        if geo:
            try:
                if header == 'Distance to Coast':
                    start_time = time.time()
                    r = distance_coast( locate[['Latitude','Longitude']].values.tolist(), *load_coast_data() )
                    st.toast(f'Calculated {header} time: {(time.time()-start_time):.2f}s', icon='⏰')

                    locate = pd.concat([locate, pd.DataFrame.from_dict({'Latitude':r['Distance_Coast']['nearestCoords(lat,lng)'][0], 'Longitude':r['Distance_Coast']['nearestCoords(lat,lng)'][1]}, orient='index').T], ignore_index=True)
                    locate['Distance'] = ['', f"{r['Distance_Coast']['kilometre']:.3f}km"]
                    with main_tab.expander('- Details -', expanded=True):
                        if address:
                            st.write(f'Address:')
                            st.info(address)
                            st.write(f'Coordinates (Latitude,Longitude):')
                            st.info(f"""
                                {r['queryCoords(lat,lng)'][0]:.6f},
                                {r['queryCoords(lat,lng)'][1]:.6f}
                                """)

                        st.write('Distance to Coast:')
                        st.info(f"{r['Distance_Coast']['kilometre']:.3f}km")
                        st.write(f'Coast Coordinates (Latitude,Longitude):')
                        st.info(f"""
                            {r['Distance_Coast']['nearestCoords(lat,lng)'][0]:.6f},
                            {r['Distance_Coast']['nearestCoords(lat,lng)'][1]:.6f}
                            """)

                elif header == 'Elevation':
                    lat, lon = locate.iloc[0][['Latitude','Longitude']]
                    start_time = time.time()
                    url = f"{HOST_v1v2}/api/{'v2' if version=='v3' else version}/elevation/?key={YOUR_API_KEY}&latlng={lat},{lon}"
                    r = requests.get(url).json()
                    st.toast(f'Calculated {header} time: {(time.time()-start_time):.2f}s', icon='⏰')

                    locate['Elevation'] = [f"{r['meters']}m"]
                    with main_tab.expander('- Details -', expanded=True):
                        if address:
                            st.write(f'Address:')
                            st.info(address)
                            st.write(f'Coordinates (Latitude,Longitude):')
                            st.info(
                                f"{lat:.6f}, {lon:.6f}"
                                )

                        st.write('Elevation:')
                        st.info(f"{r['meters']}m")

                elif header == 'Flood Data Index':
                    if st.button('📝 Fetch', use_container_width=False):

                        match_time = time.strftime('%Y.%m.%d_%H%M',time.localtime(time.time()))
                        indexing(match_time, locate, shp_fd_products, index_fd_products, os_format, private_cart_key)
                        build_private_cart(private_cart_key, map_header_cols[4])
                        r = False

                elif header in ['Flood Zone Map']:
                    lat, lon = locate.iloc[0][['Latitude','Longitude']]

                    start_time = time.time()
                    country_continent = get_country_continent(lat, lon, shp=countries_shp)
                    if country_continent:
                        country_name, continent_name = country_continent[0][0].replace(' ', ''), country_continent[0][1].replace(' ', '')

                        with st.spinner(f"Building flood zone ..."):

                            if   flood_map_size in ['1.5*1 km']: offset = 0.0075; length = 0.3045; level=16
                            elif flood_map_size in ['3*2 km']:   offset = 0.015;  length = 0.3035; level=15
                            elif flood_map_size in ['5*3 km']:   offset = 0.025;  length = 0.303;  level=14
                            elif flood_map_size in ['10*6 km']:  offset = 0.05;   length = 1;      level=13
                            elif flood_map_size in ['50*30 km']: offset = 0.25;   length = 1;      level=11

                            flood_nearest_distance = ''
                            tmp_colors = []

                            if flood_map_aris:
                                for i, flood_map_ari in enumerate(flood_map_aris):
                                    tmp = get_xarray_from_AUS_GFPlain_tif_tile(
                                        lat, lon,
                                        offset=offset,
                                        overlap=i/10-flood_map_ari,
                                        no_data=np.nan,
                                        no_nan_range=[flood_map_ari, flood_map_ari],
                                        path=f'{flood_scheme_path}{os.sep}{country_name}',
                                        ari=flood_map_ari
                                        )

                                    # # output maps
                                    # savetxt(f'ari{flood_map_ari}.asc',*nc2xr(tmp,nodata=-9999))

                                    if isinstance(tmp, xr.DataArray):

                                        tmp = tmp.expand_dims({'time': [0]})
                                        flood_coords = tmp.to_dataframe().reset_index().dropna()[['lat', 'lon']].values

                                        if i == 0:
                                            # da = tmp.copy()
                                            da = tmp.interp(
                                            	lat=np.arange(lat-offset, lat+offset, abs(tmp.lat[0]-tmp.lat[1]))[::-1],
                                            	lon=np.arange(lon-offset, lon+offset, abs(tmp.lon[0]-tmp.lon[1]))
                                            	)
                                        else:
                                            if isinstance(da, xr.DataArray):
                                                tmp = tmp.interp(lat=da.lat, lon=da.lon)
                                                da = xr.where(np.isnan(da), tmp, da)
                                            else:
                                                da = tmp.copy()
                                                da.loc[{'lat':tmp.lat[0],'lon':tmp.lon[i-1]}] = i#(i-1)/10

                                        if tmp.within:
                                            flood_nearest_distance += f'In ARI {flood_map_aris[i]} flood zone; '
                                            tmp_colors.append(flood_map_cmap.colors[i])
                                        else:
                                            if np.isnan(flood_coords).all():
                                                da = [(lat + offset, lon - offset), (lat - offset, lon + offset)]
                                                flood_nearest_distance += f'No ARI {flood_map_aris[i]} data found in the mapping area. '
                                            else:
                                                nearest_distance = get_nearest_distance(flood_coords, [(lat, lon)])
                                                flood_nearest_distance += f"Distance to ARI {flood_map_aris[i]} flood zone: {int(nearest_distance['kilometre'] * 1000)} m; "
                                                tmp_colors.append(flood_map_cmap.colors[i])
                                    else:
                                        if i == 0:
                                            da = [(lat + offset, lon - offset), (lat - offset, lon + offset)]
                                        flood_nearest_distance += f'The mapping area is not in the ARI {flood_map_aris[i]} flood zone. '

                                if np.isnan(da).all():
                                    da = [(lat + offset, lon - offset), (lat - offset, lon + offset)]

                                if isinstance(da, xr.DataArray) and len(set([i for i in da.values.flatten() if ~np.isnan(i)]))==1:
                                    tmp_colors = tmp_colors[0]

                                # def plot(lat, lng, zoom=10, map_type='roadmap', bokeh_width=500, bokeh_height=300):
                                #     from bokeh.io import show
                                #     from bokeh.plotting import gmap
                                #     from bokeh.models import GMapOptions
                                #
                                #     gmap_options = GMapOptions(lat=lat, lng=lng,
                                #                                map_type=map_type, zoom=zoom)
                                #     p = gmap("AIzaSyBo1qxaORxZijUlx-dQEkoQzHeBuIeu9WI", gmap_options, title='Pays de Gex',
                                #              width=bokeh_width, height=bokeh_height)
                                #     center = p.circle([lng], [lat], size=10, alpha=0.5, color='red')
                                #     #show(p)
                                #     return p
                                #
                                # flood_map = plot(lat, lon, map_type='terrain')

                                flood_map = add_geomap_to_doc_2(
                                    da, shape_feature=None, x='lon', y='lat', vmin=None, vmax=None, level=level,
                                    length=length, alpha=0.6, col='time', col_wrap=1, cols=1, figx=5, markersize=0,
                                    width=6309360, offset=0, cmap=ListedColormap(tmp_colors), bbox=None, style=flood_map_type
                                    )

                        st.toast(f'randered {header} time: {(time.time()-start_time):.2f}s', icon='⏰')

                elif header in ['Climate Indicators Data']:
                    if version in ['v3']:
                        if st.button('📝 Export', use_container_width=False):

                            # export_time = time.strftime('%Y.%m.%d.%H.%M.%S',time.localtime(time.time()))
                            export_time = time.strftime('%Y.%m.%d_%H%M',time.localtime(time.time()))
                            exporting(export_time, locate, params, HOST_v3, index_list, tiled_theme, rank, private_cart_key)
                            build_private_cart(private_cart_key, map_header_cols[4])
                            r = False

                elif header in ['Climate Indicators Analysis', 'Climate Risk Report', 'Climate Assessment Project']:
                    if version in ['v3']:
                        if build_able:
                            st.info(f'Found {len(locate)} locations, build?')
                            if st.button('📝 Build', use_container_width=False):

                                build_time = time.strftime('%Y.%m.%d_%H%M',time.localtime(time.time()))
                                reporting(build_time, locate, report_type, params, HOST_v3, years, flood_scheme_path, tiled_theme, rank, private_cart_key)
                                build_private_cart(private_cart_key, map_header_cols[4])
                                r = False

            except: st.warning(f'⚠️ The provided "{address}" is not contained within the {header} database.')

    with st.spinner(f"Please wait..."):

        # drawing flood map
        if header in ['Flood Zone Map'] and flood_map:
            st.markdown(f'**Address**: {address} (Latitude: {lat}, Longitude: {lon})')
            st.markdown(flood_nearest_distance)

            st.image(flood_map, caption=None, use_container_width='auto')

            foot_label = f'<p><span style="font-size:13px;">Flood Zone Map : </span><span style="color:gray;font-size:13px;">{flood_map_size}</span>'
            for i, flood_map_ari in enumerate(flood_map_aris):
                foot_label += f'<span> | </span><span style="background-color:{flood_map_cmap.colors[i]};opacity:0.7;border-radius:15%;color:{flood_map_cmap.colors[i]};font-size:13px;">0 0</span><span style="color:gray;font-size:13px;"> ARI {flood_map_aris[i]} Flood Zone</span>'

            # if not within: foot_label += f'<span> | </span><span style="color:gray;font-size:13px;">Distance to ARI 500 Flood Zone: {int(flood_nearest_distance["kilometre"]*1000)} m</span>'

            foot_label += '</p>'
            st.markdown(foot_label, unsafe_allow_html=True)


        # drawing Google map
        elif map_header_cols[1].toggle('Open GoogleMap', value=True):

            show_all_address = map_header_cols[2].toggle('Show Address on Maps', value=False)
            map = load_map(st.session_state["viewport_width"], st.session_state["viewport_width"])

            if locate.values.tolist()[0] != [0.,0.]:
                if len(locate) == 1:
                    map.set_center(locate['Longitude'].values.mean().tolist(), locate['Latitude'].values.mean().tolist(), 14)
                    folium.Marker(
                        location=locate[['Latitude','Longitude']].values.tolist()[0],
                        popup=folium.Popup(address if address else locate['Address'].iloc[0] if 'Address' in locate.columns else f"{locate[['Latitude','Longitude']].iloc[0]}", show=True) if show_all_address else None,
                        tooltip=address if address else locate['Address'].iloc[0] if 'Address' in locate.columns else f"{locate[['Latitude','Longitude']].iloc[0]}",
                        icon = folium.Icon(icon='home'),
                        ).add_to(map)
                else:
                    bbox = [
                        locate['Longitude'].min(), locate['Latitude'].min(), locate['Longitude'].max(), locate['Latitude'].max()
                        ]
                    bbox = [
                        bbox[0] - (bbox[2]-bbox[0])/5,
                        bbox[1] - (bbox[3]-bbox[1])/5,
                        bbox[2] + (bbox[2]-bbox[0])/5,
                        bbox[3] + (bbox[3]-bbox[1])/5,
                        ]
                    map.zoom_to_bounds(bbox)
                    for no, row in locate.iterrows():
                        folium.Marker(
                            location=row[['Latitude','Longitude']].values.tolist(),
                            popup=folium.Popup(row['Address'] if 'Address' in locate.columns else f"{row[['Latitude','Longitude']]}", show=True) if show_all_address else None,
                            tooltip=row['Address'] if 'Address' in locate.columns else f"{row[['Latitude','Longitude']]}",
                            icon = folium.Icon(icon='home'),
                            ).add_to(map)

            if header == 'Climate Indicators Data' and len(locate) == 1 and r:
                st.subheader(f'🏠 Location')
                st.info(f"{address} - ( Latitude: {locate[['Latitude','Longitude']].round(6).values.tolist()[0][0]}, Longitude: {locate[['Latitude','Longitude']].round(6).values.tolist()[0][1]} )")

                st.subheader("🌏 Climate Zone")
                with st.expander(list(r['KG_Climate_Classification'].values())[0]['metric'][1]):
                    st.caption(list(r['KG_Climate_Classification'].values())[0]['metric'][2])

                st.subheader("🏚 Earthquake Hazard")
                with st.expander(list(r['Earthquake_Intensity'].values())[0][0]):
                    for i in list(r['Earthquake_Intensity'].values())[0][1:]:
                        st.caption(i)

                st.subheader("🌪 Cyclone Hazard")
                with st.expander(list(r['Cyclone_Intensity'].values())[0][0]):
                    for i in list(r['Cyclone_Intensity'].values())[0][1:] + list(r['Cyclone_Frequency'].values())[0]:
                        st.caption(i)

                st.subheader("🌊 Tsunami Hazard")
                with st.expander(list(r['Tsunami_Maximum_Inundation_Height'].values())[0][0]):
                    for i in list(r['Tsunami_Maximum_Inundation_Height'].values())[0][1:]:
                        st.caption(i)

                st.subheader(f'🎓 {dict(v1="CMIP5", v2="CMIP6")[version]} variables risk score ({nscore} levels)')
                st.table(score)

                icons = {
                    'Aridity Index':'🏜',
                    'Monthly Mean Precipitation':'☔️',
                    'Monthly Mean Temperature':'🌡',
                    'Monthly Relative Humidity':'💦',
                    'Extreme Precipitation':'⛈',
                    'Extreme Temperature Days Higher 35C':'🥵',
                    'Extreme Temperature Days Lower 2C':'🥶',
                    'Extreme Temperature Days Higher 35degC':'🥵',
                    'Extreme Temperature Days Lower 2degC':'🥶',
                    'Maximum Temperature Days Higher 35degC':'🥵',
                    'Minimum Temperature Days Lower 2degC':'🥶',
                    'Extreme Wind Speed':'🌪',
                    'Heat Wave Days':'☀️',
                    'Air Heatwave Days':'☀️',
                    'Heat Stress Risk':'🩸',
                    'Cooling Degree Days':'❄️',
                    'Heating Degree Days':'♨️',
                    'Mean Sea Level Rise':'🏝',
                    'Extreme Water Level':'🌊',
                    'KBDI Fire Risk':'🔥',
                    }

                for vn in units.keys():
                    icon = icons[vn] if vn in icons else '🌈'
                    df = pd.concat([data[data.index==vn][columns].apply(pd.Series) for columns in data.columns])
                    df.index = data.columns

                    cf = pd.concat([change[change.index==vn][columns].apply(pd.Series) for columns in change.columns])
                    cf.index = change.columns

                    col11, col12 = st.columns(2)
                    if len(df.columns) == 1:
                        with col11:
                            st.subheader(f'{icon} {vn} ({units[vn]})')
                            df.columns = ['']
                            st.bar_chart(df)

                        with col12:
                            st.subheader(f'{icon} {vn} Change ({change_units[vn]})')
                            cf.columns = ['']
                            st.area_chart(cf)
                    else:
                        with col11:
                            st.subheader(f'{icon} {vn} ({units[vn]})')
                            df.columns = month_numb
                            st.line_chart(df.T)

                        with col12:
                            st.subheader(f'{icon} {vn} Change ({change_units[vn]})')
                            cf.columns = month_numb
                            st.area_chart(cf.T)

            elif header == 'Elevation' and len(locate) == 1 and r:
                map.set_center(locate['Longitude'].values.tolist()[0], locate['Latitude'].values.tolist()[0], 11)
                folium.Marker(
                    location=locate[['Latitude','Longitude']].values.tolist()[0],
                    popup=folium.Popup(f"<b>Elevation:</b> {locate['Elevation'][0]}", show=True),
                    tooltip=address,
                    icon = folium.Icon(icon='home'),
                    ).add_to(map)

            elif header == 'Distance to Coast' and len(locate) == 2:
                bbox = [
                    locate['Longitude'].min(), locate['Latitude'].min(), locate['Longitude'].max(), locate['Latitude'].max()
                    ]
                bbox = [
                    bbox[0] - (bbox[2]-bbox[0])/3,
                    bbox[1] - (bbox[3]-bbox[1])/3,
                    bbox[2] + (bbox[2]-bbox[0])/3,
                    bbox[3] + (bbox[3]-bbox[1])/3,
                    ]

                map.zoom_to_bounds(bbox)

                folium.Marker(
                    location=locate[['Latitude','Longitude']].values.tolist()[0],
                    popup=folium.Popup(address if address else locate['Address'].iloc[0] if 'Address' in locate.columns else f"{locate[['Latitude','Longitude']].iloc[0]}", show=True) if show_all_address else None,
                    tooltip=address if address else locate['Address'].iloc[0] if 'Address' in locate.columns else f"{locate[['Latitude','Longitude']].iloc[0]}",
                    icon = folium.Icon(icon='home'),
                    ).add_to(map)
                folium.Marker(
                    location=locate[['Latitude','Longitude']].values.tolist()[1],
                    popup=folium.Popup('Coast Point', show=True) if show_all_address else None,
                    tooltip=f"<b>Latitude:</b> {locate['Latitude'].values.tolist()[1]:.6f}{os.linesep}<b>Longitude:</b> {locate['Longitude'].values.tolist()[1]:.6f}",
                    icon = folium.Icon(icon='flag'),
                    ).add_to(map)
                folium.vector_layers.PolyLine(
                    locate[['Latitude','Longitude']].values.tolist(),
                    popup=folium.Popup(f"<b>Distance:</b> {locate['Distance'][1]}", show=True),
                    tooltip=locate['Distance'][1],
                    ).add_to(map)

            map.to_streamlit()

    with main_tab.expander('- About -', expanded=True):
        st.success(about[header])

####################################################################################

if __name__ == '__main__':

    #########################################################
    # Global Initialization

    streamlit_js_eval(js_expressions="window.innerWidth", key="viewport_width")

    about = {
        'ClimateHub Data Services Platform': "Embark on a groundbreaking journey with the world's leading climate and environmental data services platform — ClimateHub Data APIs. Focused on delivering unparalleled analysis and assessment data, our platform revolutionizes how we comprehend and navigate the complexities of climate and environmental data.",
        'Customized Reporting System': "We proudly present the globally foremost automated analysis and assessment report system in the field of climate and environment. In this rapidly evolving world, understanding and addressing climate change and environmental challenges are crucial. Our system provides you with accurate, in-depth insights through advanced technology and data analysis, helping you make informed decisions.",
        'Climate Toolbox (API v3.0)': "We provide the Climate Data, Risk Report, FloodZone Map, Elevation, Distance to Coast services for most location of world.",
        }

    api_urls = {
        "API for CSRD":            'http://192.168.1.136:8888',
        "API for Projects":        'http://192.168.1.65:8888',
        "API for ClimateInsights": 'http://192.168.1.16:8888',
         }
    api_index = {
        'http://192.168.1.136:8888':  ["climate_zone","elevation","earthquake","cyclone","tsunami","water_stress"],
        'http://192.168.1.65:8888':   ["climate_zone","elevation","earthquake","cyclone","tsunami","water_stress"],
        'http://192.168.1.16:8888':   ["climate_zone","elevation","earthquake","cyclone","tsunami"],
         }

    #########################################################
    # Session Initialization

    if 'countries_shp' not in st.session_state:
        st.toast('loading countries.shp ...')
        st.session_state['countries_shp'] = load_shape_data(f'CRA_Parameter{os.sep}countries.dbf')

    try:
        _module_exist_
    except:
        st.session_state[f'Jpkg_state'] = st.session_state[f'Jpkg_state'] if 'Jpkg_state' in st.session_state else load_lib_so('Jpkg')
        exec(st.session_state['Jpkg_state'])
    try:
        report_engine
    except:
        st.session_state[f'Jreport_state'] = st.session_state[f'Jreport_state'] if 'Jreport_state' in st.session_state else load_lib_so('Jreport')
        exec(st.session_state['Jreport_state'])

    #########################################################
    # logo in

    if os.path.exists(f'CRA_Parameter{os.sep}config.yaml'):
 
        with open(f'CRA_Parameter{os.sep}config.yaml') as file:
            config = yaml.safe_load(file)

        authenticator = stauth.Authenticate(
            config['credentials'],
            config['cookie']['name'],
            config['cookie']['key'],
            config['cookie']['expiry_days'],
        )

        if st.session_state['authentication_status']:
        
            tabs = st.sidebar.tabs(['Main', 'Reset password'])
            with tabs[0]:
                authenticator.logout("Logout", 'sidebar')
                st.info(f":clap: Welcome, **_{st.session_state['name']}_**")
            with tabs[1]:
                try:
                    st.info(f":clap: Welcome, **_{st.session_state['name']}_**")
                    if authenticator.reset_password(st.session_state["username"]):
                        st.success(':sunglasses: Password modified successfully')
                        with open(f'CRA_Parameter{os.sep}config.yaml', 'w') as file:
                            yaml.dump(config, file, default_flow_style=False)
                except Exception as e:
                    st.error(f'⚠️ {e}')
 
            if not st.session_state['name'] is None:
                try:
                    page_names_to_funcs = {
                        'Climate Toolbox (API v3.0)': Climate_Toolbox,
                        }

                    selected_page = tabs[0].selectbox("Select a function:", page_names_to_funcs.keys(), disabled=len(page_names_to_funcs)<=1)
                    page_names_to_funcs[selected_page](
                        HOST_v1v2 = 'http://192.168.1.16:8888',
                        HOST_v3   = 'http://192.168.1.16:9888',
                        API_KEY   = config['cookie']['key'],
                        EMAIL     = config['credentials']['usernames'][st.session_state["username"]]['email'],
                        LEVEL     = config['credentials']['usernames'][st.session_state["username"]]['roles'],
                        main_tab = tabs[0],
                        )

                except Exception as e:
                   st.session_state['authentication_status'] = False
                   st.error(f'⚠️ Error code: {e}')

        elif not st.session_state['authentication_status']:
        
            tabs = st.sidebar.tabs(['Login', 'Register', 'Username', 'Password'])
            with tabs[0]:
                authenticator.login('main')
                if st.session_state['authentication_status'] == False:
                    st.error('⚠️ Username/password is incorrect')
            with tabs[1]:
                try:
                    if authenticator.register_user('main')[0]:
                        st.success(':sunglasses: Registration submitted successfully, please send the registration information to johnny@climsystems.com for review of authority.')
                        with open(f'CRA_Parameter{os.sep}config.yaml', 'w') as file:
                            yaml.dump(config, file, default_flow_style=False)
                except Exception as e:
                    st.error(f'⚠️ {e}')
            with tabs[2]:
                try:
                    username_of_forgotten_username, \
                    email_of_forgotten_username = authenticator.forgot_username()
                    if username_of_forgotten_username:
                        st.success(f':sunglasses: Username: {username_of_forgotten_username}')
                    elif username_of_forgotten_username == False:
                        st.error('⚠️ Email not found')
                except Exception as e:
                    st.error(f'⚠️ {e}')
            with tabs[3]:
                try:
                    username_of_forgotten_password, \
                    email_of_forgotten_password, \
                    new_random_password = authenticator.forgot_password()
                    if username_of_forgotten_password:
                        st.success(':sunglasses: New password to be sent securely.')
                    elif username_of_forgotten_password == False:
                        st.error('⚠️ Username not found')
                except Exception as e:
                    st.error(f'⚠️ {e}')

    else:
        st.error('⚠️ Host config is not been found, please report error details to johnny@climsystems.com for support.')

####################################################################################
