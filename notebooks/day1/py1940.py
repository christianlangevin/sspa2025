
import pathlib as pl
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import folium
import flopy


def get_crs():
    return "EPSG:2229"


def files_from_namefile(namefile_path):
    """
    Get a dictionary of file names from the name file.
    """
    from flopy.utils import mfreadnam
    ext_unit_dict = mfreadnam.parsenamefile(
        namefile_path, {}, verbose=True
    )
    fname_dict = {name_entry.filetype.lower(): name_entry.filename for name_entry in ext_unit_dict.values()}
    return fname_dict


def get_isarm_gridprops():
    nlay = 5
    nrow = 1642
    ncol = 2243
    dx = 102.53
    dy = 102.53
    xll = 6605135.735254  # from nwt name file
    yll = 1798429.241150  # from nwt name file
    rotation = -27.05  # from nwt name file

    delr = np.empty((ncol), dtype=float)
    delr.fill(dx)
    delc = np.empty((nrow), dtype=float)
    delc.fill(dy)

    gridprops = {
        "nlay": nlay,
        "nrow": nrow,
        "ncol": ncol,
        "delr": delr,
        "delc": delc,
        "xoff": xll,
        "yoff": yll,
        "angrot": rotation,
        "crs": "EPSG:2229",
    }

    return gridprops

def get_isarm_modelgrid():
    return flopy.discretization.StructuredGrid(
        **get_isarm_gridprops()
    )

def isarm_info():
    for k, v in get_isarm_gridprops().items():
        print(f"{k}: {v}")


def cutout_info():
    for k, v in get_cutout_gridprops().items():
        print(f"{k}: {v}")


def get_cutout_modelgrid():
    return flopy.discretization.StructuredGrid(
        **get_cutout_gridprops()
    )


def get_coarsened_modelgrid():
    return flopy.discretization.StructuredGrid(
        **get_coarsened_gridprops()
    )


def get_coarsened_gridprops():
    mg_isarm = get_isarm_modelgrid()

    irowstart = 32
    icolstart = 483

    # note these are the extended rows and columns required to evenly coarsen
    # the original grid by a factor of 10
    nrow_child = 600 # 597
    ncol_child = 1430 # 1426

    dx = mg_isarm.delr[0]
    dy = mg_isarm.delc[0]

    nrow = mg_isarm.nrow
    ncol = mg_isarm.ncol
    xll = mg_isarm.xoffset
    yll = mg_isarm.yoffset
    rotation = mg_isarm.angrot

    x = icolstart * dx
    y = nrow * dy - irowstart * dy - nrow_child * dy

    # find lower left coordinates of the coarsened cutout grid
    xll_child, yll_child = flopy.utils.geometry.transform(x, y, xll, yll, rotation * np.pi / 180.0)

    spatial_coarsening = 10

    # recalculate the number of rows and columns in the coarsened grid
    nrow_child = int(nrow_child / spatial_coarsening)
    ncol_child = int(ncol_child / spatial_coarsening)

    delr_child = np.empty((ncol_child), dtype=float)
    delr_child.fill(dx * spatial_coarsening)
    delc_child = np.empty((nrow_child), dtype=float)
    delc_child.fill(dy * spatial_coarsening)

    gridprops = {
        "nlay": mg_isarm.nlay,
        "nrow": nrow_child,
        "ncol": ncol_child,
        "delr": delr_child,
        "delc": delc_child,
        "xoff": xll_child,
        "yoff": yll_child,
        "angrot": rotation,
        "crs": mg_isarm.crs,
    }
    return gridprops


def get_cutout_gridprops():

    mg_isarm = get_isarm_modelgrid()

    irowstart = 32
    icolstart = 483
    nrow_child = 597
    ncol_child = 1426

    dx = mg_isarm.delr[0]
    dy = mg_isarm.delc[0]

    nrow = mg_isarm.nrow
    ncol = mg_isarm.ncol
    xll = mg_isarm.xoffset
    yll = mg_isarm.yoffset
    rotation = mg_isarm.angrot

    x = icolstart * dx
    y = nrow * dy - irowstart * dy - nrow_child * dy
    xll_child, yll_child = flopy.utils.geometry.transform(x, y, xll, yll, rotation * np.pi / 180.0)

    delr_child = np.empty((ncol_child), dtype=float)
    delr_child.fill(dx)
    delc_child = np.empty((nrow_child), dtype=float)
    delc_child.fill(dy)

    gridprops = {
        "nlay": mg_isarm.nlay,
        "nrow": nrow_child,
        "ncol": ncol_child,
        "delr": delr_child,
        "delc": delc_child,
        "xoff": xll_child,
        "yoff": yll_child,
        "angrot": rotation,
        "crs": mg_isarm.crs,
    }
    return gridprops


def get_gdf_ibound(modelgrid, ibound, layer):
    gdf = modelgrid.geo_dataframe
    gdf["ibound"] = ibound[layer].flatten()
    gdf_ibound = gdf.dissolve(by="ibound", aggfunc="sum", as_index=False)
    return gdf_ibound


def get_gdf_ibc(modelgrid, ibc):
    gdf = modelgrid.geo_dataframe
    gdf["ibc"] = ibc.flatten()
    gdf_ibc = gdf.dissolve(by="ibc", aggfunc="sum", as_index=False)
    return gdf_ibc


def get_ibc(model_ws, namefile, layer, kper, wel=True, sfr=True):
    m = flopy.modflow.Modflow.load(
        namefile,
        model_ws=model_ws,
        check=False,
        forgive=True,
        load_only=["dis", "bas6"],
        version="mfnwt",
    )
    filename_dict = files_from_namefile(model_ws / namefile)

    ibound = m.bas6.ibound.array
    ibc = ibound.copy()[layer]

    if wel or sfr:
        mtemp = flopy.modflow.Modflow()
        f = model_ws / filename_dict["dis"]
        print(f"Loading {f}")
        _ = flopy.modflow.ModflowDis.load(f, mtemp)

    if wel:
        f = model_ws / filename_dict["wel"]
        print(f"Loading {f}")
        wel = flopy.modflow.ModflowWel.load(f, mtemp, nper=max(1, kper))
        spd = wel.stress_period_data[kper]
        idx = np.where(spd["k"] == layer)[0]
        i = spd[idx]["i"]
        j = spd[idx]["j"]
        ibc[i, j] = 2

    if sfr:
        f = model_ws / filename_dict["sfr"]
        print(f"Loading {f}")
        sfr = flopy.modflow.ModflowSfr2.load(f, mtemp, nper=1)
        i = sfr.reach_data["i"]
        j = sfr.reach_data["j"]
        ibc[i, j] = 3

    return ibc


def plot_hfb(ax, modelgrid, layer, hfb_data, **plotkwargs):
    xvertices = modelgrid.xvertices
    yvertices = modelgrid.yvertices
    for row in hfb_data:
        k, r1, c1, r2, c2, hydchr = row
        if k != layer:
            continue

        # Find the shared edge between (r1, c1) and (r2, c2)
        if r1 == r2:
            # Same row, different columns: vertical edge
            row = r1
            col = min(c1, c2) + 1
            x = [xvertices[row, col], xvertices[row + 1, col]]
            y = [yvertices[row, col], yvertices[row + 1, col]]
        elif c1 == c2:
            # Same column, different rows: horizontal edge
            col = c1
            row = min(r1, r2) + 1
            x = [xvertices[row, col], xvertices[row, col + 1]]
            y = [yvertices[row, col], yvertices[row, col + 1]]
        else:
            # Not adjacent (should not happen for HFB)
            continue

        ax.plot(x, y, **plotkwargs)


def hfb_to_gpd(modelgrid, hfb_data, layer=None):
    """
    Convert HFB data to a GeoDataFrame.
    
    Parameters:
    - modelgrid: flopy discretization StructuredGrid object.
    - hfb_data: List of tuples containing HFB data (layer, r1, c1, r2, c2, hydchr).
    
    Returns:
    - gdf: GeoDataFrame with HFB lines.
    """
    from shapely.geometry import LineString

    lines = []
    hydchr_list = []
    layer_list = []
    xvertices = modelgrid.xvertices
    yvertices = modelgrid.yvertices
    if layer is None:
        layer = list(np.unique(hfb_data["k"]))
    else:
        layer = [layer] if isinstance(layer, int) else layer


    for row in hfb_data:
        k, r1, c1, r2, c2, hydchr = row
        if k in layer:

            # Find the shared edge between (r1, c1) and (r2, c2)
            if r1 == r2:
                # Same row, different columns: vertical edge
                row = r1
                col = min(c1, c2) + 1
                x = [xvertices[row, col], xvertices[row + 1, col]]
                y = [yvertices[row, col], yvertices[row + 1, col]]
            elif c1 == c2:
                # Same column, different rows: horizontal edge
                col = c1
                row = min(r1, r2) + 1
                x = [xvertices[row, col], xvertices[row, col + 1]]
                y = [yvertices[row, col], yvertices[row, col + 1]]
            else:
                # Not adjacent (should not happen for HFB)
                continue

            line = LineString(zip(x, y))
            lines.append(line)
            hydchr_list.append(hydchr)
            layer_list.append(k)

    gdf = gpd.GeoDataFrame(geometry=lines)
    gdf["hydchr"] = hydchr_list
    gdf["layer"] = layer_list
    gdf.crs = modelgrid.crs
    return gdf

