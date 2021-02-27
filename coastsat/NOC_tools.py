from coastsat.SDS_tools import *


def polygon_from_kml(fn):
    """
    Extracts coordinates from a .kml file.

    KV WRL 2018

    Arguments:
    -----------
    fn: str
        file_path + filename of the kml file to be read

    Returns:
    -----------
    polygon: list
        coordinates extracted from the .kml file

    """

    # read .kml file
    with open(fn) as kmlFile:
        doc = kmlFile.read()
        # parse to find coordinates field
    str1 = '<coordinates>'
    str2 = '</coordinates>'
    subdoc = doc[doc.find(str1) + len(str1):doc.find(str2)]
    coordlist = subdoc.split('\n')

    coordlist = subdoc.split(',')
    # read coordinates
    polygon = []

    for coord_pair in range(int(len(coordlist) / 2)):
        lat = float(coordlist[coord_pair * 2].replace('0 ', ''))
        lon = float(coordlist[coord_pair * 2 + 1].replace('0 ', ''))
        polygon.append([lat, lon])

    # read coordinates
    #    polygon = []
    #    for i in range(1, len(coordlist) - 1):
    #        polygon.append([float(coordlist[i].split(',')[0]), float(coordlist[i].split(',')[1])])

    return [polygon]


def output_to_gdf(shorelines, metadata):

    gdf_all = None
    for index, shoreline in enumerate(shorelines):

        if len(shoreline) == 0:
            continue
        else:
            geom = geometry.LineString(shoreline)
            gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(geom))
            gdf.index = [index]
            gdf.loc[index, 'sat_name'] = metadata['sat_name']
            gdf.loc[index, 'date_start'] = output['date_start']
            gdf.loc[index, 'date_end'] = output['date_end']
            gdf.loc[index, 'number_images'] = output['number_images']

            if index == 0:
                gdf_all = gdf
            else:
                gdf_all = gdf_all.append(gdf)

    return gdf_all
