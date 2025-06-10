# ---------------------------------------------------------------------------
# Reporting function
# Input: parameters, netCDF data
# Output: DOCX, PDF
# ---------------------------------------------------------------------------
# Author: Zhaoquan YU - Johnny
# Create: 20/11/2022
# Last Modify: 10/06/2025
# ---------------------------------------------------------------------------

###########################################################
# reporting engine
###########################################################

def report_engine(report_type, path, info, years, flood_scheme_path, **kwargs):
    report_type = report_type.replace(' ', '_')
    return eval( f"{report_type}(path, info, years, flood_scheme_path, **kwargs)" )

###########################################################
# reporting generator
###########################################################

def CSRD_Climate_Indicators_Analysis(path, info, years, flood_scheme_path):
    '''
    report_type: title()
    '''

    root = 'CRA_report'
    report_name = 'CSRD_Climate_Indicators_Analysis'
    input_path = output_path = path

    #########################################################

    left  = .21
    right = .95
    width = 6309360 *.95 / 2
    figsize = (5,2.8)

    hrs = 24
    ari = 100
    depth = 7
    nscore = 10
    pths = [5,50,95]
    ssps = [126,245,585]

    #########################################################

    info = info.astype(str)
    messages = []
    all_sites_summary_table = {}
    all_sites_summary_table_score = {}

    for sn, row in info.iterrows():

        if not set(['Latitude', 'Longitude', 'Address', 'Report type']) <= set(info.columns):
            return {'INFO':'The uploaded file • CSV must contains "Latitude, Longitude, Address, Report type" columns.'}

        latlng_list = list(zip( [float(i) for i in row['Latitude'].split(',')], [float(i) for i in row['Longitude'].split(',')] ))
        lat, lon = latlng_list[0]
        site_name = row['Address'].translate(fileNameTranslationTable)
        report_type = row['Report type'].strip().title()
        country_name, continent_name = get_country_continent(lat, lon, st.session_state['countries_shp'])[0]
        country_name, continent_name = country_name.replace(' ', ''), continent_name.replace(' ', '')

        print(f'SN: {(sn+1):02d}_{site_name} - {report_type}')

        #########################################################

        print('Loading Maps ......')
        map1 = add_geomap_to_doc_2(latlng_list, shape_feature=None, x='lon', y='lat', vmin=None, vmax=None, level=11, length=5.,
            col='time', col_wrap=1, cols=1, figx=5, markersize=50, width=6309360/2, offset=0.15, cmap=None, bbox=None, style='satellite')
        map2 = add_geomap_to_doc_2(latlng_list, shape_feature=None, x='lon', y='lat', vmin=None, vmax=None, level=18, length=0.1,
            col='time', col_wrap=1, cols=1, figx=5, markersize=200, width=6309360/2, offset=0.003, cmap=None, bbox=None, style='satellite')

        #########################################################
        # generate data dicts

        vns = []

        table_dict = {}
        boxplot_dict = {}
        errorbar_dict = {}

        mon_table_dict = {}
        mon_boxplot_dict = {}
        mon_errorbar_dict = {}

        season_table_dict = {}
        season_boxplot_dict = {}
        season_errorbar_dict = {}

        if glob.glob(f'{input_path}{os.sep}*Data.nc'): # per process v3 netCDF
            data = xr.open_dataset(glob.glob(f'{input_path}{os.sep}*Data.nc')[0]).squeeze(drop=True).fillna(0)
            score = xr.open_dataset(glob.glob(f'{input_path}{os.sep}*Score.nc')[0]).squeeze(drop=True).fillna(0)
            change = xr.open_dataset(glob.glob(f'{input_path}{os.sep}*Change.nc')[0]).squeeze(drop=True).fillna(0)

            if 'ari' in data.coords:
                data =  data.sel(ari=100, drop=True)
            if 'hrs' in data.coords:
                data =  data.sel(hrs=24, drop=True)
            if 'ari' in score.coords:
                score = score.sel(ari=100, drop=True)
            if 'hrs' in score.coords:
                score = score.sel(hrs=24, drop=True)
            if 'ari' in change.coords:
                change =change.sel(ari=100, drop=True)
            if 'hrs' in change.coords:
                change =change.sel(hrs=24, drop=True)

            for _vn in data.data_vars:
                vn = _vn.replace('_',' ')
                data = data.rename({_vn: vn})
                score = score.rename({_vn: vn})
                change = change.rename({_vn: vn})
                units[vn] = data[vn].units
                change_units[vn] = change[vn].units
                if 'cat' in data[vn].coords:
                    data[vn] = data[vn].sel(cat=cat_calculate[vn]).sum('cat', skipna=True, keep_attrs=True) if vn in cat_calculate else data[vn].sum('cat', skipna=True, keep_attrs=True)
                if 'cat' in score[vn].coords:
                    score[vn] = score[vn].sel(cat=cat_calculate[vn]).mean('cat', skipna=True, keep_attrs=True) if vn in cat_calculate else score[vn].mean('cat', skipna=True, keep_attrs=True)
                if 'cat' in change[vn].coords:
                    change[vn] = change[vn].sel(cat=cat_calculate[vn]).sum('cat', skipna=True, keep_attrs=True) if vn in cat_calculate else change[vn].sum('cat', skipna=True, keep_attrs=True)
            data = data.drop('cat', errors='ignore')
            score = score.drop('cat', errors='ignore')
            change = change.drop('cat', errors='ignore')

            ssps = data.ssp.values
            years = np.setdiff1d(data.year.astype(str).values, '2005').tolist()
            n_ssp, n_year = len(ssps), len(years) + 1

            try:
                information = eval(data.Information)[sn]
                details = [
                    ['Location', f"{site_name} - ( Latitude: {latlng_list[0][0]:.6f}, Longitude: {latlng_list[0][1]:.6f} )"],
                    ["Elevation", '%d %s' % list(information['Elevation'].items())[0][::-1] ],
                    ]
                for i in list(information['KG_Climate_Classification'].values())[0]['metric']:
                    details.append(["Climate Zone", i])
                for i in list(information['Earthquake_Intensity'].values())[0]:
                    details.append(["Earthquake", i])
                for i in list(information['Cyclone_Intensity'].values())[0] + list(information['Cyclone_Frequency'].values())[0]:
                    details.append(["Cyclone", i])
                for i in list(information['Tsunami_Maximum_Inundation_Height'].values())[0]:
                    details.append(["Tsunami", i])
            except Exception as error:
                information = {}
                details = [
                    ['Location', f"{site_name} - ( Latitude: {latlng_list[0][0]:.6f}, Longitude: {latlng_list[0][1]:.6f} )"],
                    ]
            details = pd.DataFrame(details, columns=['', 'Details']).set_index('')

        for ssp in ssps:

            if glob.glob(f'{input_path}{os.sep}*Data.nc'): # convert v3 netCDF to dict

                if 'point' in data.coords:
                    df05 = {
                        'Information': details,
                        'Climate Risk Score': score.sel(point=sn, ssp=ssp, pth=5, method='nearest').to_dataframe()[list(data.data_vars)].T
                    }
                    df50 = {
                        'Information': details,
                        'Climate Risk Score': score.sel(point=sn, ssp=ssp, pth=50, method='nearest').to_dataframe()[list(data.data_vars)].T
                    }
                    df95 = {
                        'Information': details,
                        'Climate Risk Score': score.sel(point=sn, ssp=ssp, pth=95, method='nearest').to_dataframe()[list(data.data_vars)].T
                    }
                    for vn in data.data_vars:
                        if 'month' in data[vn].coords:
                            df05[vn] = pd.concat([
                                data[vn].sel(point=sn, ssp=ssp, pth=5, method='nearest').to_dataframe()[vn].unstack(level=0), pd.DataFrame(index=['nan'] * 2),
                                change[vn].sel(point=sn, ssp=ssp, pth=5, method='nearest').to_dataframe()[vn].unstack(level=0)])
                            df50[vn] = pd.concat([
                                data[vn].sel(point=sn, ssp=ssp, pth=50, method='nearest').to_dataframe()[vn].unstack(level=0), pd.DataFrame(index=['nan'] * 2),
                                change[vn].sel(point=sn, ssp=ssp, pth=50, method='nearest').to_dataframe()[vn].unstack(level=0)])
                            df95[vn] = pd.concat([
                                data[vn].sel(point=sn, ssp=ssp, pth=95, method='nearest').to_dataframe()[vn].unstack(level=0), pd.DataFrame(index=['nan'] * 2),
                                change[vn].sel(point=sn, ssp=ssp, pth=95, method='nearest').to_dataframe()[vn].unstack(level=0)])
                            df05[vn].index.name = df50[vn].index.name = df95[vn].index.name = 'year'
                        else:
                            df05[vn] = pd.concat([
                                data[vn].sel(point=sn, ssp=ssp, pth=5, method='nearest').to_dataframe()[vn],
                                change[vn].sel(point=sn, ssp=ssp, pth=5, method='nearest').to_dataframe()[vn]], axis=1)
                            df50[vn] = pd.concat([
                                data[vn].sel(point=sn, ssp=ssp, pth=50, method='nearest').to_dataframe()[vn],
                                change[vn].sel(point=sn, ssp=ssp, pth=50, method='nearest').to_dataframe()[vn]], axis=1)
                            df95[vn] = pd.concat([
                                data[vn].sel(point=sn, ssp=ssp, pth=95, method='nearest').to_dataframe()[vn],
                                change[vn].sel(point=sn, ssp=ssp, pth=95, method='nearest').to_dataframe()[vn]], axis=1)
                else:
                    df05 = {
                        'Information': details,
                        'Climate Risk Score': score.sel(ssp=ssp, pth=5, method='nearest').to_dataframe()[list(data.data_vars)].T
                    }
                    df50 = {
                        'Information': details,
                        'Climate Risk Score': score.sel(ssp=ssp, pth=50, method='nearest').to_dataframe()[list(data.data_vars)].T
                    }
                    df95 = {
                        'Information': details,
                        'Climate Risk Score': score.sel(ssp=ssp, pth=95, method='nearest').to_dataframe()[list(data.data_vars)].T
                    }
                    for vn in data.data_vars:
                        if 'month' in data[vn].coords:
                            df05[vn] = pd.concat([
                                data[vn].sel(ssp=ssp, pth=5, method='nearest').to_dataframe()[vn].unstack(level=0), pd.DataFrame(index=['nan'] * 2),
                                change[vn].sel(ssp=ssp, pth=5, method='nearest').to_dataframe()[vn].unstack(level=0)])
                            df50[vn] = pd.concat([
                                data[vn].sel(ssp=ssp, pth=50, method='nearest').to_dataframe()[vn].unstack(level=0), pd.DataFrame(index=['nan'] * 2),
                                change[vn].sel(ssp=ssp, pth=50, method='nearest').to_dataframe()[vn].unstack(level=0)])
                            df95[vn] = pd.concat([
                                data[vn].sel(ssp=ssp, pth=95, method='nearest').to_dataframe()[vn].unstack(level=0), pd.DataFrame(index=['nan'] * 2),
                                change[vn].sel(ssp=ssp, pth=95, method='nearest').to_dataframe()[vn].unstack(level=0)])
                            df05[vn].index.name = df50[vn].index.name = df95[vn].index.name = 'year'
                        else:
                            df05[vn] = pd.concat([
                                data[vn].sel(ssp=ssp, pth=5, method='nearest').to_dataframe()[vn],
                                change[vn].sel(ssp=ssp, pth=5, method='nearest').to_dataframe()[vn]], axis=1)
                            df50[vn] = pd.concat([
                                data[vn].sel(ssp=ssp, pth=50, method='nearest').to_dataframe()[vn],
                                change[vn].sel(ssp=ssp, pth=50, method='nearest').to_dataframe()[vn]], axis=1)
                            df95[vn] = pd.concat([
                                data[vn].sel(ssp=ssp, pth=95, method='nearest').to_dataframe()[vn],
                                change[vn].sel(ssp=ssp, pth=95, method='nearest').to_dataframe()[vn]], axis=1)

                df05 = { k: convert2str_index_columns_baseline(df05[k]) for k in df05 }
                df50 = { k: convert2str_index_columns_baseline(df50[k]) for k in df50 }
                df95 = { k: convert2str_index_columns_baseline(df95[k]) for k in df95 }

            elif glob.glob(f'{input_path}{os.sep}*.xlsx'): # convert v2 Excel to dict

                df05 = pd.read_excel(
                    glob.glob(f'{input_path}{os.sep}*{site_name} - (CMIP6, ssp={ssp}, ari={ari}, hrs={hrs}, pth=5, depth={depth}, Risk level={nscore}).xlsx')[0],
                    sheet_name=None, index_col=0)
                df05 = { k: convert2str_index_columns_baseline(df05[k]) for k in df05 }
                df50 = pd.read_excel(
                    glob.glob(f'{input_path}{os.sep}*{site_name} - (CMIP6, ssp={ssp}, ari={ari}, hrs={hrs}, pth=50, depth={depth}, Risk level={nscore}).xlsx')[0],
                    sheet_name=None, index_col=0)
                df50 = { k: convert2str_index_columns_baseline(df50[k]) for k in df50 }
                df95 = pd.read_excel(
                    glob.glob(f'{input_path}{os.sep}*{site_name} - (CMIP6, ssp={ssp}, ari={ari}, hrs={hrs}, pth=95, depth={depth}, Risk level={nscore}).xlsx')[0],
                    sheet_name=None, index_col=0)
                df95 = { k: convert2str_index_columns_baseline(df95[k]) for k in df95 }

            else: print({'INFO': 'Not find data files.'})

            for sheet_name in df50: # transform dataframe
                print('Processing', ssp, sheet_name)
                if sheet_name in ['Information']:
                    df = df50[sheet_name]
                    if sheet_name in table_dict:
                        table_dict[sheet_name].append( df )
                    else:
                        table_dict[sheet_name] = [ df ]

                elif sheet_name in ['Climate Risk Score']:
                    df = (
                        df50[sheet_name].map(score_format) + ' (' +
                        df05[sheet_name].map(score_format) + ',' +
                        df95[sheet_name].map(score_format) + ')'
                        )
               
                    df.columns = [i if i=='Baseline' else f'SSP{ISO_name[ssp]} - {i}' for i in df.columns]
                    if 'Baseline' in df.columns: df['Baseline'] = df['Baseline'].map(dropDuplicates_floatString)
 
                    if sheet_name in table_dict:
                        table_dict[sheet_name].append( df )
                    else:
                        table_dict[sheet_name] = [ df ]

                    #########################################################

                    boxplot_df = pd.concat([ # Baseline + future values
                        df05[sheet_name],
                        df50[sheet_name],
                        df95[sheet_name]
                        ], axis=0)
                    boxplot_df.columns = [i if i=='Baseline' else f'SSP{ISO_name[ssp]} - {i}' for i in boxplot_df.columns]
                    if sheet_name in boxplot_dict:
                        boxplot_dict[sheet_name].append( boxplot_df )
                    else:
                        boxplot_dict[sheet_name] = [ boxplot_df ]

                    #########################################################

                    df05[sheet_name].columns = df50[sheet_name].columns = df95[sheet_name].columns = [i if i=='Baseline' else f'SSP{ISO_name[ssp]} - {i}' for i in df05[sheet_name].columns]
                    if sheet_name in errorbar_dict:
                        errorbar_dict[sheet_name].append( {5:df05[sheet_name], 50:df50[sheet_name], 95:df95[sheet_name]} )
                    else:
                        errorbar_dict[sheet_name] = [ {5:df05[sheet_name], 50:df50[sheet_name], 95:df95[sheet_name]} ]

                elif 'nan' in df50[sheet_name].index: # monthly variables

                    if not np.all(df50[sheet_name].iloc[:len(years)+1]==0) and sheet_name not in vns: vns.append(sheet_name)

                    if 1: # generate monthly table, boxplot, errorbar

                        df = ( # Baseline + future values
                            df50[sheet_name].iloc[:len(years)+1].applymap("{:.02f}".format) + ' (' +
                            df05[sheet_name].iloc[:len(years)+1].applymap("{:.02f}".format) + ', ' +
                            df95[sheet_name].iloc[:len(years)+1].applymap("{:.02f}".format) + ')'
                            ).replace('nan (nan, nan)', '0.00 (0.00, 0.00)')

                        if 1:
                            tmp = ( # Baseline + future changes
                                df50[sheet_name].iloc[len(years)+3:].applymap("{:.02f}".format) + ' (' +
                                df05[sheet_name].iloc[len(years)+3:].applymap("{:.02f}".format) + ', ' +
                                df95[sheet_name].iloc[len(years)+3:].applymap("{:.02f}".format) + ')'
                                ).replace('nan (nan, nan)', '0.00 (0.00, 0.00)')
                            tmp.iloc[0, :] = df.iloc[0, :]
                            df = tmp.copy()

                        df.index = [i if i == 'Baseline' else f'SSP{ISO_name[ssp]} - {i}' for i in df.index]
                        if 'Baseline' in df.index: df.loc['Baseline'] = df.loc['Baseline'].map(dropDuplicates_floatString)

                        if sheet_name in mon_table_dict:
                            mon_table_dict[sheet_name].append( df )
                        else:
                            mon_table_dict[sheet_name] = [ df ]

                        #########################################################

                        boxplot_df = pd.concat([ # changes
                            df05[sheet_name].iloc[len(years)+3:],
                            df50[sheet_name].iloc[len(years)+3:],
                            df95[sheet_name].iloc[len(years)+3:]
                            ], axis=1)
                        boxplot_df['SSP'] = f'SSP{ISO_name[ssp]}'
                        if sheet_name in mon_boxplot_dict:
                            mon_boxplot_dict[sheet_name].append( boxplot_df )
                        else:
                            mon_boxplot_dict[sheet_name] = [ boxplot_df ]

                        #########################################################

                        tmp05, tmp50, tmp95 = df05[sheet_name].copy(), df50[sheet_name].copy(), df95[sheet_name].copy()
                        tmp05['SSP'] = tmp50['SSP'] = tmp95['SSP'] = f'SSP{ISO_name[ssp]}'
                        if sheet_name in mon_errorbar_dict: # projections
                            mon_errorbar_dict[sheet_name].append( {5:tmp05.iloc[:len(years)+1], 50:tmp50.iloc[:len(years)+1], 95:tmp95.iloc[:len(years)+1]} )
                        else:
                            mon_errorbar_dict[sheet_name] = [ {5:tmp05.iloc[:len(years)+1], 50:tmp50.iloc[:len(years)+1], 95:tmp95.iloc[:len(years)+1]} ]

                    if 2: # generate seasonal table, boxplot, errorbar

                        if sheet_name in accumulate_vns:
                            tmp50 = pd.concat([ df50[sheet_name][season_to_month[season]].sum(axis=1) for season in season_to_month ], axis=1)
                            tmp05 = pd.concat([ df05[sheet_name][season_to_month[season]].sum(axis=1) for season in season_to_month ], axis=1)
                            tmp95 = pd.concat([ df95[sheet_name][season_to_month[season]].sum(axis=1) for season in season_to_month ], axis=1)
                        else:
                            tmp50 = pd.concat([ df50[sheet_name][season_to_month[season]].mean(axis=1) for season in season_to_month ], axis=1)
                            tmp05 = pd.concat([ df05[sheet_name][season_to_month[season]].mean(axis=1) for season in season_to_month ], axis=1)
                            tmp95 = pd.concat([ df95[sheet_name][season_to_month[season]].mean(axis=1) for season in season_to_month ], axis=1)

                        tmp05.columns = tmp50.columns = tmp95.columns = list(season_to_month.keys())

                        df = ( # Baseline + future values
                            tmp50.iloc[:len(years)+1].applymap("{:.02f}".format) + ' (' +
                            tmp05.iloc[:len(years)+1].applymap("{:.02f}".format) + ', ' +
                            tmp95.iloc[:len(years)+1].applymap("{:.02f}".format) + ')'
                            ).replace('nan (nan, nan)', '0.00 (0.00, 0.00)')

                        if sheet_name in anomaly_vns: # changeV
                            tmp = (
                                (tmp50.iloc[:len(years)+1] - tmp50.iloc[0]).applymap("{:.02f}".format) + ' (' +
                                (tmp05.iloc[:len(years)+1] - tmp05.iloc[0]).applymap("{:.02f}".format) + ', ' +
                                (tmp95.iloc[:len(years)+1] - tmp95.iloc[0]).applymap("{:.02f}".format) + ')'
                                ).replace('nan (nan, nan)', '0.00 (0.00, 0.00)')
                            boxplot_df = pd.concat([
                                tmp50.iloc[:len(years)+1] - tmp50.iloc[0],
                                tmp05.iloc[:len(years)+1] - tmp05.iloc[0],
                                tmp95.iloc[:len(years)+1] - tmp95.iloc[0]
                                ], axis=1)
                        else: # changeP
                            tmp = (
                                (tmp50.iloc[:len(years)+1] / tmp50.iloc[0] * 100 - 100).applymap("{:.02f}".format) + ' (' +
                                (tmp05.iloc[:len(years)+1] / tmp05.iloc[0] * 100 - 100).applymap("{:.02f}".format) + ', ' +
                                (tmp95.iloc[:len(years)+1] / tmp95.iloc[0] * 100 - 100).applymap("{:.02f}".format) + ')'
                                ).replace('nan (nan, nan)', '0.00 (0.00, 0.00)')
                            boxplot_df = pd.concat([
                                tmp50.iloc[:len(years)+1] / tmp50.iloc[0] * 100 - 100,
                                tmp05.iloc[:len(years)+1] / tmp05.iloc[0] * 100 - 100,
                                tmp95.iloc[:len(years)+1] / tmp95.iloc[0] * 100 - 100
                                ], axis=1)

                        df = pd.concat([df, tmp], axis=1)
                        df.iloc[0, len(season_to_month):] = df.iloc[0, :len(season_to_month)]

                        if 0:
                            df = df.iloc[:, :len(season_to_month)]  # Baseline + future values
                        else:
                            df = df.iloc[:, len(season_to_month):]  # Baseline + future changes

                        df.index = [i if i == 'Baseline' else f'SSP{ISO_name[ssp]} - {i}' for i in df.index]
                        if 'Baseline' in df.index: df.loc['Baseline'] = df.loc['Baseline'].map(dropDuplicates_floatString)

                        if sheet_name in season_table_dict:
                            season_table_dict[sheet_name].append( df )
                        else:
                            season_table_dict[sheet_name] = [ df ]

                        #########################################################

                        boxplot_df['SSP'] = f'SSP{ISO_name[ssp]}'
                        if sheet_name in season_boxplot_dict:
                            season_boxplot_dict[sheet_name].append( boxplot_df )
                        else:
                            season_boxplot_dict[sheet_name] = [ boxplot_df ]

                        #########################################################

                        tmp05['SSP'] = tmp50['SSP'] = tmp95['SSP'] = f'SSP{ISO_name[ssp]}'
                        if sheet_name in season_errorbar_dict: # projections
                            season_errorbar_dict[sheet_name].append( {5:tmp05.iloc[:len(years)+1], 50:tmp50.iloc[:len(years)+1], 95:tmp95.iloc[:len(years)+1]} )
                        else:
                            season_errorbar_dict[sheet_name] = [ {5:tmp05.iloc[:len(years)+1], 50:tmp50.iloc[:len(years)+1], 95:tmp95.iloc[:len(years)+1]} ]

                    if 3: # generate annual table, boxplot, errorbar

                        if sheet_name in accumulate_vns:
                            tmp50 = df50[sheet_name].sum(axis=1)
                            tmp05 = df05[sheet_name].sum(axis=1)
                            tmp95 = df95[sheet_name].sum(axis=1)
                        else:
                            tmp50 = df50[sheet_name].mean(axis=1)
                            tmp05 = df05[sheet_name].mean(axis=1)
                            tmp95 = df95[sheet_name].mean(axis=1)

                        df = ( # Baseline + future values
                            tmp50.iloc[:len(years)+1].to_frame().applymap("{:.02f}".format) + ' (' +
                            tmp05.iloc[:len(years)+1].to_frame().applymap("{:.02f}".format) + ', ' +
                            tmp95.iloc[:len(years)+1].to_frame().applymap("{:.02f}".format) + ')'
                            ).replace('nan (nan, nan)', '0.00 (0.00, 0.00)')


                        if sheet_name in anomaly_vns: # changeV
                            tmp = (
                                (tmp50.iloc[:len(years)+1] - tmp50.iloc[0]).to_frame().applymap("{:.02f}".format) + ' (' +
                                (tmp05.iloc[:len(years)+1] - tmp05.iloc[0]).to_frame().applymap("{:.02f}".format) + ', ' +
                                (tmp95.iloc[:len(years)+1] - tmp95.iloc[0]).to_frame().applymap("{:.02f}".format) + ')'
                                ).replace('nan (nan, nan)', '0.00 (0.00, 0.00)')
                            boxplot_df = pd.concat([
                                tmp50.iloc[:len(years)+1] - tmp50.iloc[0],
                                tmp05.iloc[:len(years)+1] - tmp05.iloc[0],
                                tmp95.iloc[:len(years)+1] - tmp95.iloc[0]
                                ], axis=0).to_frame()
                        else: # changeP
                            tmp = (
                                (tmp50.iloc[:len(years)+1] / tmp50.iloc[0] * 100 - 100).to_frame().applymap("{:.02f}".format) + ' (' +
                                (tmp05.iloc[:len(years)+1] / tmp05.iloc[0] * 100 - 100).to_frame().applymap("{:.02f}".format) + ', ' +
                                (tmp95.iloc[:len(years)+1] / tmp95.iloc[0] * 100 - 100).to_frame().applymap("{:.02f}".format) + ')'
                                ).replace('nan (nan, nan)', '0.00 (0.00, 0.00)')
                            boxplot_df = pd.concat([
                                tmp50.iloc[:len(years)+1] / tmp50.iloc[0] * 100 - 100,
                                tmp05.iloc[:len(years)+1] / tmp05.iloc[0] * 100 - 100,
                                tmp95.iloc[:len(years)+1] / tmp95.iloc[0] * 100 - 100
                                ], axis=0).to_frame()

                        df = pd.concat([df, tmp], axis=1)
                        df.columns = ['value', 'change']
                        df[df.index.get_level_values('year').str.contains('Baseline')] = df[df.index.get_level_values('year').str.contains('Baseline')].map(dropDuplicates_floatString)
                        df.loc[df.index.get_level_values('year').str.contains('Baseline'), 'change'] = df.loc[df.index.get_level_values('year').str.contains('Baseline'), 'value']

                        if 0:
                            df = df.iloc[:, 0].to_frame()  # Baseline + future values
                        else:
                            df = df.iloc[:, 1].to_frame()  # Baseline + future changes

                        df.columns = [f'SSP{ISO_name[ssp]}']

                        if sheet_name in table_dict:
                            table_dict[sheet_name].append( df )
                        else:
                            table_dict[sheet_name] = [ df ]

                        #########################################################

                        boxplot_df.columns = [f'SSP{ISO_name[ssp]}']
                        if sheet_name in boxplot_dict:
                            boxplot_dict[sheet_name].append( boxplot_df )
                        else:
                            boxplot_dict[sheet_name] = [ boxplot_df ]

                        #########################################################

                        if sheet_name in errorbar_dict: # projections
                            errorbar_dict[sheet_name].append( {5:tmp05.iloc[:len(years)+1].to_frame(f'SSP{ISO_name[ssp]}'), 50:tmp50.iloc[:len(years)+1].to_frame(f'SSP{ISO_name[ssp]}'), 95:tmp95.iloc[:len(years)+1].to_frame(f'SSP{ISO_name[ssp]}')} )
                        else:
                            errorbar_dict[sheet_name] = [ {5:tmp05.iloc[:len(years)+1].to_frame(f'SSP{ISO_name[ssp]}'), 50:tmp50.iloc[:len(years)+1].to_frame(f'SSP{ISO_name[ssp]}'), 95:tmp95.iloc[:len(years)+1].to_frame(f'SSP{ISO_name[ssp]}')} ]

                else: # annual variables
                    if not np.all(df50[sheet_name]==0) and sheet_name not in vns: vns.append(sheet_name)
                    df = (
                        df50[sheet_name].applymap("{:.02f}".format) + ' (' +
                        df05[sheet_name].applymap("{:.02f}".format) + ', ' +
                        df95[sheet_name].applymap("{:.02f}".format) + ')'
                        ).replace('nan (nan, nan)', '0.00 (0.00, 0.00)')

                    df.columns = ['value', 'change']
                    df[df.index.get_level_values('year').str.contains('Baseline')] = df[df.index.get_level_values('year').str.contains('Baseline')].map(dropDuplicates_floatString)
                    df.loc[df.index.get_level_values('year').str.contains('Baseline'), 'change'] = df.loc[df.index.get_level_values('year').str.contains('Baseline'), 'value']

                    if 0:
                        df = df.iloc[:,0].to_frame() # Baseline + future values
                    else:
                        df = df.iloc[:,1].to_frame() # Baseline + future changes

                    df.columns = [f'SSP{ISO_name[ssp]}']

                    if sheet_name in table_dict:
                        table_dict[sheet_name].append( df )
                    else:
                        table_dict[sheet_name] = [ df ]

                    #########################################################

                    boxplot_df = pd.concat([ # changes
                        df05[sheet_name],
                        df50[sheet_name],
                        df95[sheet_name]
                        ], axis=0).iloc[:,1].to_frame()
                    boxplot_df.columns = [f'SSP{ISO_name[ssp]}']

                    if sheet_name in boxplot_dict:
                        boxplot_dict[sheet_name].append( boxplot_df )
                    else:
                        boxplot_dict[sheet_name] = [ boxplot_df ]

                    #########################################################

                    df05[sheet_name].columns = df50[sheet_name].columns = df95[sheet_name].columns = [f'SSP{ISO_name[ssp]}' for i in df05[sheet_name].columns]
                    if sheet_name in errorbar_dict: # projections
                        errorbar_dict[sheet_name].append( {5:df05[sheet_name].iloc[:,0].to_frame(), 50:df50[sheet_name].iloc[:,0].to_frame(), 95:df95[sheet_name].iloc[:,0].to_frame()} )
                    else:
                        errorbar_dict[sheet_name] = [ {5:df05[sheet_name].iloc[:,0].to_frame(), 50:df50[sheet_name].iloc[:,0].to_frame(), 95:df95[sheet_name].iloc[:,0].to_frame()} ]


        #########################################################
        # creat Document


        #////////////////////////////////////////////////////////
        vns = [i for i in ['Climate Zone','Elevation','Earthquake','Tsunami','Cyclone'] if i in table_dict['Information'][0].index] + ['Rainfall Flood Depth'] + vns
        #////////////////////////////////////////////////////////


        doc, vns = generate_template(report_name, report_type[:31], vns, years, root)

        address_list = site_name.split(os.linesep)
        address_list = [address_list[i] if i<len(address_list) else '' for i in range(3)]

        vns = [f'({i+1}) {vns[i]}' for i in range(len(vns))]

        doc = add_text_to_doc(
            report_type,
            doc,
            placeholder='placeholder.report_type',
            )
        doc = add_text_to_doc(
            f'{datetime.date.today():%d/%m/%Y}',
            doc,
            placeholder='placeholder.report_date',
            )
        doc = add_text_to_doc(
            f"(Latitude: {latlng_list[0][0]:.6f}, Longitude: {latlng_list[0][1]:.6f})",
            # f"(Latitude: {latlng_list[0][0]:.6f}, Longitude: {latlng_list[0][1]:.6f}, Elevation: {df05['Information'].loc['Elevation','Details']})",
            doc,
            placeholder='placeholder.latlng.text',
            size=Pt(10), bold=False, font=font,
            )
        doc = add_text_to_doc(
            os.linesep.join( address_list ),
            doc,
            placeholder='placeholder.address.list',
            size=Pt(10), bold=False, font=font,
            )
        doc = add_text_to_doc(
            os.linesep.join( vns ),
            doc,
            placeholder='placeholder.variable.list',
            size=Pt(10), bold=False, font=font,
            )

        map_tag = re.compile(r'%\(placeholder.address.map\)s')
        flood_tag1 = re.compile(r'%\(placeholder.flood.map1\)s')
        flood_tag2 = re.compile(r'%\(placeholder.flood.map2\)s')
        flood_tag3 = re.compile(r'%\(placeholder.flood.map3\)s')

        line_break = re.compile(r'%\(placeholder.line_break\)s')
        page_break = re.compile(r'%\(placeholder.page_break\)s')

        for i, p in enumerate(doc.paragraphs):
            if bool(page_break.match(p.text)):
                temp_text = page_break.split(p.text)
                p.runs[0].text = temp_text[0]
                p.add_run().add_break(WD_BREAK.PAGE)
                p.paragraph_format.space_after = 0
            elif bool(line_break.match(p.text)):
                temp_text = line_break.split(p.text)
                p.runs[0].text = temp_text[0]
            elif bool(map_tag.match(p.text)):
                for _p in p.runs: _p.text = ''
                p.add_run().add_picture(map1, width=width)
                p.add_run(' ')
                p.add_run().add_picture(map2, width=width)
            elif bool(flood_tag1.match(p.text)):
                for _p in p.runs: _p.text = ''
                p.add_run().add_picture(flood_map1, width=width*1.9)
            elif bool(flood_tag2.match(p.text)):
                for _p in p.runs: _p.text = ''
                p.add_run().add_picture(flood_map2, width=width*1.9)
            elif bool(flood_tag3.match(p.text)):
                for _p in p.runs: _p.text = ''
                p.add_run().add_picture(flood_map3, width=width*1.9)

        ofn = f'{output_path}{os.sep}SN{(sn+1):02d}_{site_name[:30]}...{report_name}.{report_type}.docx'.replace(' ','_')

        #########################################################
        # drawing
        for sheet_name in table_dict:

            #########################################################
            # tables
            print('Generating table:', sheet_name)

            # output index tables
            if sheet_name in ['Information']:
                table = table_dict[sheet_name][0].replace(np.nan, '')
                table_name = f'Table: {sheet_name}'

                for key in set(table.index):
                    if key in ['Climate Zone']:
                        index_text = os.linesep.join([i[0] for i in table[table.index==key].values.tolist()[1:] if i[0]])
                        if index_text.split(os.linesep)[0]=='Ocean': messages.append(sn)
                    else:
                        index_text = os.linesep.join([i[0] for i in table[table.index==key].values.tolist() if i[0]])

                    doc = add_text_to_doc(
                        index_text,
                        doc,
                        placeholder=f'placeholder.{key}.text',
                        # color='#2874A6',
                        bold=True,
                        # italic=True
                        )

            # output score tables
            elif sheet_name in ['Climate Risk Score']:
                table = [table_dict[sheet_name][i].iloc[:,1:] if i>0 else table_dict[sheet_name][i] for i in range(len(table_dict[sheet_name]))]
                table = pd.concat(table,axis=1)
                table.insert(0,'',table.index)
                table_name = f"Table: {sheet_name} for the Baseline period and future risk score (1-{nscore} levels)"

                doc = add_text_to_doc(
                    table_name,
                    doc,
                    placeholder=f'placeholder.{sheet_name}.summary.table.text',
                    size=Pt(10), bold=False, alignment='left'
                    )
                doc = add_table_to_doc(
                    table.replace([0, '0', '0 (0,0)'], '-N/A-'),
                    doc,
                    placeholder=f'placeholder.{sheet_name}.summary.table',
                    font=font, fontsize=table_fontsize, rowheight=rowheight, cell_alignment=cell_alignment,
                    color=score_table_color, row0_fontcolor=score_table_header_fontcolor, col0_fontcolor=score_table_header_fontcolor,
                    row_heat_colors=row_heat_colors
                    )

            # output annual tables
            else:
                table = pd.concat(table_dict[sheet_name],axis=1)

                if 'ari' in data[sheet_name].coords and 'hrs' in data[sheet_name].coords:
                    tmp = table.loc[str(ari),str(hrs)]
                elif 'ari' in data[sheet_name].coords:
                    tmp = table.loc[str(ari)]
                elif 'hrs' in data[sheet_name].coords:
                    tmp = table.loc[str(hrs)]
                else: tmp = table.copy()

                tmp = tmp.unstack(0).to_frame().T
                tmp.columns = ['Baseline' if x[1] in ['Baseline','2005',2005] else f'{x[0]} {x[1]}' for x in tmp.columns]
                tmp = pd.concat([
                    tmp[['Baseline']].T.drop_duplicates().T,
                    tmp[tmp.columns.drop('Baseline')]
                    ], axis=1)
                tmp.insert(0, 'Address', site_name)

                if sheet_name in all_sites_summary_table:
                    all_sites_summary_table[sheet_name].append( tmp )
                else:
                    all_sites_summary_table[sheet_name] = [ tmp ]

                table.reset_index(inplace=True)
                table.rename(columns={'ari':'ARI', 'hrs':'Duration (hrs)', 'year':'Year'}, inplace=True)
                table_name = f"Table: {sheet_name.replace('Monthly','Annual')} Baseline ({units[sheet_name]}) and Future Changes ({change_units[sheet_name]})"

            doc = add_text_to_doc(
                table_name,
                doc,
                placeholder=f'placeholder.{sheet_name}.annual.table.text',
                size=Pt(10), bold=False, alignment='left'
                )
            doc = add_table_to_doc(
                table,
                doc,
                placeholder=f'placeholder.{sheet_name}.annual.table',
                font=font, fontsize=table_fontsize, rowheight=rowheight, cell_alignment=cell_alignment,
                color=table_color, row0_fontcolor=table_header_fontcolor, col0_fontcolor=table_header_fontcolor,
                row_heat_colors=row_heat_colors if sheet_name in ['Climate Risk Score'] else None
                )

            # output monthly tables
            if sheet_name in mon_table_dict:
                table = [mon_table_dict[sheet_name][i].iloc[1:,:] if i>0 else mon_table_dict[sheet_name][i] for i in range(len(mon_table_dict[sheet_name]))]
                table = pd.concat(table,axis=0).T
                table.reset_index(inplace=True, names='Month')
                table_name = f"Table: {sheet_name} Baseline ({units[sheet_name]}) and Future Changes ({change_units[sheet_name]})"

                doc = add_text_to_doc(
                    table_name,
                    doc,
                    placeholder=f'placeholder.{sheet_name}.mon.table.text',
                    size=Pt(10), bold=False, alignment='left'
                    )
                doc = add_table_to_doc(
                    table,
                    doc,
                    placeholder=f'placeholder.{sheet_name}.mon.table',
                    font=font, fontsize=table_fontsize, rowheight=rowheight, cell_alignment=cell_alignment,
                    color=table_color, row0_fontcolor=table_header_fontcolor, col0_fontcolor=table_header_fontcolor,
                    row_heat_colors=row_heat_colors if sheet_name in ['Climate Risk Score'] else None
                    )

            # output seasonal tables
            if sheet_name in season_table_dict:
                table = [season_table_dict[sheet_name][i].iloc[1:,:] if i>0 else season_table_dict[sheet_name][i] for i in range(len(season_table_dict[sheet_name]))]
                table = pd.concat(table,axis=0).T
                table.reset_index(inplace=True, names='Season')
                table_name = f"Table: {sheet_name.replace('Monthly','Seasonal')} Baseline ({units[sheet_name]}) and Future Changes ({change_units[sheet_name]})"

                doc = add_text_to_doc(
                    table_name,
                    doc,
                    placeholder=f'placeholder.{sheet_name}.season.table.text',
                    size=Pt(10), bold=False, alignment='left'
                    )
                doc = add_table_to_doc(
                    table,
                    doc,
                    placeholder=f'placeholder.{sheet_name}.season.table',
                    font=font, fontsize=table_fontsize, rowheight=rowheight, cell_alignment=cell_alignment,
                    color=table_color, row0_fontcolor=table_header_fontcolor, col0_fontcolor=table_header_fontcolor,
                    row_heat_colors=row_heat_colors if sheet_name in ['Climate Risk Score'] else None
                    )

            #########################################################
            # errorbars
            if sheet_name in errorbar_dict:

                if sheet_name in ['Climate Risk Score']: continue

                print('Generating errorbar:', sheet_name)

                bars_DataFrame = pd.concat( [i[50] for i in errorbar_dict[sheet_name]], axis=1 )
                yerr05 = pd.concat( [i[5] for i in errorbar_dict[sheet_name]], axis=1 )
                yerr95 = pd.concat( [i[95] for i in errorbar_dict[sheet_name]], axis=1 )

                if 'ari' in data[sheet_name].coords and 'hrs' in data[sheet_name].coords:
                    bars_DataFrame = bars_DataFrame.loc[str(ari),str(hrs)]
                    yerr05 = yerr05.loc[str(ari),str(hrs)]
                    yerr95 = yerr95.loc[str(ari),str(hrs)]
                elif 'ari' in data[sheet_name].coords:
                    bars_DataFrame = bars_DataFrame.loc[str(ari)]
                    yerr05 = yerr05.loc[str(ari)]
                    yerr95 = yerr95.loc[str(ari)]
                elif 'hrs' in data[sheet_name].coords:
                    bars_DataFrame = bars_DataFrame.loc[str(hrs)]
                    yerr05 = yerr05.loc[str(hrs)]
                    yerr95 = yerr95.loc[str(hrs)]

                ymax = pd.concat([bars_DataFrame, yerr05, yerr95]).astype('float32').max(numeric_only=True).max()
                ymin = pd.concat([bars_DataFrame, yerr05, yerr95]).astype('float32').min(numeric_only=True).min()
                top = ymax + abs(ymax-ymin)*0.15
                bottom = ymin - abs(ymax-ymin)*0.1

                if top-bottom<1: top, bottom = top*1.5, top/2

                yerr05 = bars_DataFrame - yerr05
                yerr95 = yerr95 - bars_DataFrame
                yerr = np.stack((yerr05.T, yerr95.T), axis=1).clip(0)

                doc = add_errorbar_to_doc(
                    bars_DataFrame=bars_DataFrame,
                    yerr=yerr,
                    doc_object=doc,
                    placeholder=f'placeholder.{sheet_name}.errorbar',
                    title=f'{vn_abbr[sheet_name] if sheet_name in vn_abbr else sheet_name} Projections',
                    xlabel='Years',
                    ylabel=f"{vn_abbr[sheet_name] if sheet_name in vn_abbr else sheet_name}{os.linesep}({units[sheet_name]})",
                    xticks=None,
                    xgridstyle='',
                    ygridstyle='--',
                    cmaps=box_colors[sheet_name] if sheet_name in box_colors else ['#DAF7A6', '#FFC300', '#FF5733'],
                    figsize=figsize,
                    fontsize=10,
                    title_fontsize=12,
                    bar_width=.8,
                    width=width,
                    left=left,
                    right=right,
                    ymax=top,
                    ymin=bottom,
                    xrot=0,
                    yrot=0,
                    )

            if sheet_name in mon_errorbar_dict:

                print('Generating mon_errorbar:', sheet_name)

                baseline_dict = {}
                for pth in [5,50,95]:
                    tmp_list = []
                    for table in mon_errorbar_dict[sheet_name]:
                        ssp = table[pth]['SSP'][0]
                        tmp = table[pth][table[pth].index.str.contains('Baseline')].T
                        tmp.columns = ['Baseline']
                        tmp_list.append(tmp)
                    table = pd.concat(tmp_list, axis=1)
                    table = table[table.index!='SSP']
                    baseline_dict[pth] = table

                for year in years:

                    pth_dict = {}
                    for pth in [5,50,95]:
                        tmp_list = []
                        for table in mon_errorbar_dict[sheet_name]:
                            ssp = table[pth]['SSP'][0]
                            tmp = table[pth][table[pth].index.str.contains(year)].T
                            tmp.columns = [ssp]
                            tmp_list.append(tmp)
                        table = pd.concat(tmp_list, axis=1)
                        table = table[table.index!='SSP']
                        table.insert(0, 'Baseline', baseline_dict[pth].iloc[:,0])
                        pth_dict[pth] = table

                    bars_DataFrame = pth_dict[50]
                    yerr05 = pth_dict[5]
                    yerr95 = pth_dict[95]
                    ymax = pd.concat([bars_DataFrame, yerr05, yerr95]).astype('float32').max(numeric_only=True).max()
                    ymin = pd.concat([bars_DataFrame, yerr05, yerr95]).astype('float32').min(numeric_only=True).min()
                    top = ymax + abs(ymax-ymin)*0.15
                    bottom = ymin - abs(ymax-ymin)*0.1

                    if top-bottom<1: top, bottom = top*1.5, top/2

                    yerr05 = bars_DataFrame - yerr05
                    yerr95 = yerr95 - bars_DataFrame
                    yerr = np.stack((yerr05.T, yerr95.T), axis=1).clip(0)

                    doc = add_errorbar_to_doc(
                        bars_DataFrame=bars_DataFrame,
                        yerr=yerr,
                        doc_object=doc,
                        placeholder=f'placeholder.{sheet_name}.mon.errorbar.{year}',
                        title=f'{vn_abbr[sheet_name] if sheet_name in vn_abbr else sheet_name} Projections {year}',
                        xlabel='Months',
                        ylabel=f"{vn_abbr[sheet_name] if sheet_name in vn_abbr else sheet_name}{os.linesep}({units[sheet_name]})",
                        xticks=None,
                        xgridstyle='',
                        ygridstyle='--',
                        cmaps=box_colors[sheet_name] if sheet_name in box_colors else ['#DAF7A6', '#FFC300', '#FF5733'],
                        figsize=figsize,
                        fontsize=10,
                        title_fontsize=12,
                        bar_width=.8,
                        width=width,
                        left=left,
                        right=right,
                        ymax=top,
                        ymin=bottom,
                        xrot=0,
                        yrot=0,
                        )

            if sheet_name in season_errorbar_dict:

                print('Generating season_errorbar:', sheet_name)

                baseline_dict = {}
                for pth in [5,50,95]:
                    tmp_list = []
                    for table in season_errorbar_dict[sheet_name]:
                        ssp = table[pth]['SSP'][0]
                        tmp = table[pth][table[pth].index.str.contains('Baseline')].T
                        tmp.columns = ['Baseline']
                        tmp_list.append(tmp)
                    table = pd.concat(tmp_list, axis=1)
                    table = table[table.index!='SSP']
                    baseline_dict[pth] = table

                for year in years:

                    pth_dict = {}
                    for pth in [5,50,95]:
                        tmp_list = []
                        for table in season_errorbar_dict[sheet_name]:
                            ssp = table[pth]['SSP'][0]
                            tmp = table[pth][table[pth].index.str.contains(year)].T
                            tmp.columns = [ssp]
                            tmp_list.append(tmp)
                        table = pd.concat(tmp_list, axis=1)
                        table = table[table.index!='SSP']
                        table.insert(0, 'Baseline', baseline_dict[pth].iloc[:,0])
                        pth_dict[pth] = table

                    bars_DataFrame = pth_dict[50]
                    yerr05 = pth_dict[5]
                    yerr95 = pth_dict[95]
                    ymax = pd.concat([bars_DataFrame, yerr05, yerr95]).astype('float32').max(numeric_only=True).max()
                    ymin = pd.concat([bars_DataFrame, yerr05, yerr95]).astype('float32').min(numeric_only=True).min()
                    top = ymax + abs(ymax-ymin)*0.15
                    bottom = ymin - abs(ymax-ymin)*0.1

                    if top-bottom<1: top, bottom = top*1.5, top/2

                    yerr05 = bars_DataFrame - yerr05
                    yerr95 = yerr95 - bars_DataFrame
                    yerr = np.stack((yerr05.T, yerr95.T), axis=1).clip(0)

                    doc = add_errorbar_to_doc(
                        bars_DataFrame=bars_DataFrame,
                        yerr=yerr,
                        doc_object=doc,
                        placeholder=f'placeholder.{sheet_name}.season.errorbar.{year}',
                        title=f'{vn_abbr[sheet_name] if sheet_name in vn_abbr else sheet_name} Projections {year}',
                        xlabel='Seasons',
                        ylabel=f"{vn_abbr[sheet_name] if sheet_name in vn_abbr else sheet_name}{os.linesep}({units[sheet_name]})",
                        xticks=None,
                        xgridstyle='',
                        ygridstyle='--',
                        cmaps=box_colors[sheet_name] if sheet_name in box_colors else ['#DAF7A6', '#FFC300', '#FF5733'],
                        figsize=figsize,
                        fontsize=10,
                        title_fontsize=12,
                        bar_width=.8,
                        width=width,
                        left=left,
                        right=right,
                        ymax=top,
                        ymin=bottom,
                        xrot=0,
                        yrot=0,
                        )

            #########################################################
            # boxplots
            if sheet_name in boxplot_dict:

                if sheet_name in ['Climate Risk Score']: continue

                print('Generating boxplots:', sheet_name)

                table = pd.concat(boxplot_dict[sheet_name], axis=1)

                if 'ari' in data[sheet_name].coords and 'hrs' in data[sheet_name].coords:
                    table = table.loc[str(ari),str(hrs)]
                elif 'ari' in data[sheet_name].coords:
                    table = table.loc[str(ari)]
                elif 'hrs' in data[sheet_name].coords:
                    table = table.loc[str(hrs)]

                table = table[table.index!='Baseline']
                table.index.name = 'Years'
                columns = list(table.columns)
                table = table.groupby(table.index).quantile([0, .5, 1])
                # table = table.groupby(table.index).quantile([0, .159, .5, .841, 1])
                table = table.reset_index(level='Years')

                doc = add_boxplot_grouped_to_doc(
                    table,
                    doc_object=doc,
                    placeholder=f'placeholder.{sheet_name}.boxplot',
                    whis=(0,100), # 0Pth point shows 5Pth value(minimum in data), 100Pth point shows 95Pth value(maximun in data),
                    by='Years',
                    columns=columns,
                    title=f'{vn_abbr[sheet_name] if sheet_name in vn_abbr else sheet_name} Changes',
                    xlabel='Years',
                    ylabel=f"Changes ({change_units[sheet_name] if sheet_name in change_units else ''})",
                    legend='SSPs',
                    figsize=figsize,
                    fontsize=10,
                    title_fontsize=12,
                    cmaps=box_colors[sheet_name] if sheet_name in box_colors else ['#DAF7A6', '#FFC300', '#FF5733'],
                    box_width=0.5,
                    width=width,
                    left=left,
                    right=right,
                    )

            if sheet_name in mon_boxplot_dict:

                print('Generating mon_boxplot:', sheet_name)

                for year in years:
                    ssp_list = []
                    for table in mon_boxplot_dict[sheet_name]:
                        ssp = table['SSP'][0]
                        table = table[table.index.str.contains(year)].T
                        table.columns = [ssp]
                        ssp_list.append(table)
                    table = pd.concat(ssp_list, axis=1)
                    table = table[table.index!='SSP']
                    table.index.name = 'Months'
                    columns = list(table.columns)

                    table = table.astype(float).groupby(table.index, sort=False).quantile([0, .5, 1])
                    # table = table.astype(float).groupby(table.index, sort=False).quantile([0, .159, .5, .841, 1])
                    table = table.reset_index(level='Months')

                    doc = add_boxplot_grouped_to_doc(
                        table,
                        doc_object=doc,
                        placeholder=f'placeholder.{sheet_name}.mon.boxplot.{year}',
                        by='Months',
                        columns=columns,
                        title=f'{vn_abbr[sheet_name] if sheet_name in vn_abbr else sheet_name} Changes {year}',
                        xlabel='Months',
                        ylabel=f"Changes ({change_units[sheet_name] if sheet_name in change_units else ''})",
                        legend='SSPs',
                        figsize=figsize,
                        fontsize=10,
                        title_fontsize=12,
                        cmaps=box_colors[sheet_name] if sheet_name in box_colors else ['#DAF7A6', '#FFC300', '#FF5733'],
                        box_width=0.5,
                        width=width,
                        left=left,
                        right=right,
                        )

            if sheet_name in season_boxplot_dict:

                print('Generating season_boxplot:', sheet_name)

                for year in years:
                    ssp_list = []
                    for table in season_boxplot_dict[sheet_name]:
                        ssp = table['SSP'][0]
                        table = table[table.index.str.contains(year)].T
                        table.columns = [ssp]
                        ssp_list.append(table)
                    table = pd.concat(ssp_list, axis=1)
                    table = table[table.index!='SSP']
                    table.index.name = 'Seasons'
                    columns = list(table.columns)

                    table = table.astype(float).groupby(table.index, sort=False).quantile([0, .5, 1])
                    # table = table.astype(float).groupby(table.index, sort=False).quantile([0, .159, .5, .841, 1])
                    table = table.reset_index(level='Seasons')

                    doc = add_boxplot_grouped_to_doc(
                        table,
                        doc_object=doc,
                        placeholder=f'placeholder.{sheet_name}.season.boxplot.{year}',
                        by='Seasons',
                        columns=columns,
                        title=f'{vn_abbr[sheet_name] if sheet_name in vn_abbr else sheet_name} Changes {year}',
                        xlabel='Seasons',
                        ylabel=f"Changes ({change_units[sheet_name] if sheet_name in change_units else ''})",
                        legend='SSPs',
                        figsize=figsize,
                        fontsize=10,
                        title_fontsize=12,
                        cmaps=box_colors[sheet_name] if sheet_name in box_colors else ['#DAF7A6', '#FFC300', '#FF5733'],
                        box_width=0.5,
                        width=width,
                        left=left,
                        right=right,
                        )

            #########################################################
            # figure name
            if sheet_name in errorbar_dict and sheet_name in boxplot_dict:

                if sheet_name in ['Climate Risk Score']: continue

                print('Generating annual figure name:', sheet_name)

                if 'ari' in data[sheet_name].coords and 'hrs' in data[sheet_name].coords:
                    figure_name = f"Figure: {sheet_name.replace('Monthly','Annual')} {hrs}Hours ARI{ari} Baseline, Projections ({units[sheet_name]}) and Future Changes ({change_units[sheet_name]})."
                elif 'ari' in data[sheet_name].coords:
                    figure_name = f"Figure: {sheet_name.replace('Monthly','Annual')} ARI{ari} Baseline, Projections ({units[sheet_name]}) and Future Changes ({change_units[sheet_name]})."
                elif 'hrs' in data[sheet_name].coords:
                    figure_name = f"Figure: {sheet_name.replace('Monthly','Annual')} {hrs}Hours Baseline, Projections ({units[sheet_name]}) and Future Changes ({change_units[sheet_name]})."
                else:
                    figure_name = f"Figure: {sheet_name.replace('Monthly','Annual')} Baseline, Projections ({units[sheet_name]}) and Future Changes ({change_units[sheet_name]})."

                doc = add_text_to_doc(
                    figure_name,
                    doc,
                    placeholder=f'placeholder.{sheet_name}.figure.text',
                    size=Pt(10), bold=False, alignment='left'
                    )

            if sheet_name in mon_errorbar_dict and sheet_name in mon_boxplot_dict:

                print('Generating monthly figure name:', sheet_name)

                figure_name = f"Figure: {sheet_name} Baseline, Projections ({units[sheet_name]}) and Future Changes ({change_units[sheet_name]})."

                doc = add_text_to_doc(
                    figure_name,
                    doc,
                    placeholder=f'placeholder.{sheet_name}.mon.figure.text',
                    size=Pt(10), bold=False, alignment='left'
                    )

            if sheet_name in season_errorbar_dict and sheet_name in season_boxplot_dict:

                print('Generating monthly figure name:', sheet_name)

                figure_name = f"Figure: {sheet_name.replace('Monthly','Seasonal')} Baseline, Projections ({units[sheet_name]}) and Future Changes ({change_units[sheet_name]})."

                doc = add_text_to_doc(
                    figure_name,
                    doc,
                    placeholder=f'placeholder.{sheet_name}.season.figure.text',
                    size=Pt(10), bold=False, alignment='left'
                    )

            #########################################################
            # score table
            if sheet_name not in ['Information', 'Climate Risk Score']:
                table = [table_dict['Climate Risk Score'][i].iloc[:,1:] if i>0 else table_dict['Climate Risk Score'][i] for i in range(len(table_dict['Climate Risk Score']))]
                table = pd.concat(table,axis=1)
                table.columns = [i.replace(' - ', ' ') for i in table.columns]

                tmp = table[table.index==sheet_name].replace([0, '0', '0 (0,0)'], '-N/A-')
                tmp.insert(0, 'Address', site_name)
                if sheet_name in all_sites_summary_table_score:
                    all_sites_summary_table_score[sheet_name].append( tmp )
                else:
                    all_sites_summary_table_score[sheet_name] = [ tmp ]

                table.insert(0,'','Risk Score')
                if sheet_name in table.index:
                    doc = add_text_to_doc(
                        f"Table: Risk Scores (1-{nscore} levels)",
                        doc,
                        placeholder=f'placeholder.{sheet_name}.score.text',
                        size=Pt(10), bold=False, alignment='left'
                        )
                    doc = add_table_to_doc(
                        table[table.index==sheet_name].replace([0, '0', '0 (0,0)'], '-N/A-'),
                        doc,
                        placeholder=f'placeholder.{sheet_name}.score.table',
                        font=font, fontsize=table_fontsize, rowheight=rowheight, cell_alignment=cell_alignment,
                        color=score_table_color, row0_fontcolor=score_table_header_fontcolor, col0_fontcolor=score_table_header_fontcolor,
                        row_heat_colors=row_heat_colors,
                        )

        #########################################################

        doc.save( ofn )

    #########################################################

    doc._body.clear_content()

    for sheet_name in all_sites_summary_table:

        print(f'Generating Summary table: {sheet_name}')

        doc.add_paragraph(f"Table: Annual {sheet_name.replace('Monthly ','')} ({units[sheet_name]}) Summary.")

        table = pd.concat(all_sites_summary_table[sheet_name])

        doc = add_table_to_doc(
            table,
            doc,
            font=font, fontsize=table_fontsize, rowheight=rowheight, cell_alignment=cell_alignment,
            color=table_color, row0_fontcolor=table_header_fontcolor, col0_fontcolor=table_header_fontcolor,
            )

        doc.add_page_break()

    doc.save( f'{output_path}{os.sep}Summary_Data_Tables.docx' )

    #########################################################

    doc._body.clear_content()

    for sheet_name in all_sites_summary_table_score:

        print(f'Generating Summary score table: {sheet_name}')

        doc.add_paragraph(f"Table: Annual {sheet_name.replace('Monthly ','')} Score (1-{nscore}) Summary.")

        table = pd.concat(all_sites_summary_table_score[sheet_name])

        doc = add_table_to_doc(
            table,
            doc,
            font=font, fontsize=table_fontsize, rowheight=rowheight, cell_alignment=cell_alignment,
            color=table_color, row0_fontcolor=table_header_fontcolor, col0_fontcolor=table_header_fontcolor,
            )

        doc.add_page_break()

    doc.save( f'{output_path}{os.sep}Summary_Score_Tables.docx' )

    #########################################################

    print('Please check:', messages)
    return True



###########################################################
