import pandas as pd
import constants
from constants import *
import get_FF_data
from get_FF_data import *

"""
    ONLY USE TO MODIFY THE DATA IN THE CSVs
"""

# remove players who do not appear in consecutive seasons,
# they will be useless for the model
def filter_csvs(files, root, names, skiprows):
    data = read_csvs(files, root, names, [], skiprows)
    for i in range(len(data)):
        data[i].set_index("Player", inplace=True)
    # iterate over all of the dataframes cread in read_csvs() and remove players
    # from consecutive seasons that don't appear on both lists (didn't play the
    # previous season or the next season). Each dataframe appended to
    # configured_data holds all of the players for that season's previous season
    # stats and next seasons rank for the training data.
    for i in range(len(data)):
        for name in data[i].index.values:
            if i == 0:
                if not name in data[i+1].index.values:
                    data[i].drop(name, inplace=True)
            elif i == len(data)-1:
                if not name in data[i-1].index.values:
                    data[i].drop(name, inplace=True)
            else:
                if not name in data[i+1].index.values and not name in data[i-1].index.values:
                    data[i].drop(name, inplace=True)

    for i in range(len(data)):
        data[i].to_csv(root+files[i])
        
# one time call just to format the csvs both in a more
# readable way and with per game stats added
def configure_csvs(files, root, names, skiprows):
    cols = []
    for i in range(len(names)+1):
        if not i == 0:
            cols.append(i)
            
    for filename in files:
        filename = root + filename
        df = pd.read_csv(filename, names=names, skiprows=skiprows, usecols=cols)
        df.dropna(subset=['FantPos'], inplace=True)
        df.fillna(0, inplace=True)   # replace all NaN values with 0
        df["Player"] = df.apply(lambda x: x["Player"].split("\\")[1], axis=1)
        df.set_index("Player", inplace=True)
        
        qbtd_pg = []
        cmp_pg = []
        qbatt_pg = []
        qbyds_pg = []
        ints_pg = []
        rbatt_pg = []
        rbyds_pg = []
        rbtd_pg = []
        tgt_pg = []
        rec_pg = []
        wryds_pg = []
        wrtd_pg = []
        fmb_pg = []
        tottd_pg = []
        twopm_pg = []
        fantpt_pg = []
        ppr_pg = []
        qbypa = []
        wrppgppr = []
        wrppg = []
        rbppg = []
        qbppg = []
        qb = []
        rb= []
        wr = []
        te = []
        categories_3 = []
        categories_6 = []
        for  i in range(len(df)):
            pos = df["FantPos"].iloc[i]
            if pos == "QB":
                qb.append(1)
                rb.append(0)
                wr.append(0)
                te.append(0)
            elif pos == "RB":
                qb.append(0)
                rb.append(1)
                wr.append(0)
                te.append(0)
            elif pos == "WR":
                qb.append(0)
                rb.append(0)
                wr.append(1)
                te.append(0)
            elif pos == "TE":
                qb.append(0)
                rb.append(0)
                wr.append(0)
                te.append(1)
            gp = df["G"].iloc[i]
            qbtd = df["QBTD"].iloc[i]
            cmp = df["Cmp"].iloc[i]
            qbatt = df["QBAtt"].iloc[i]
            qbyds = df["QBYds"].iloc[i]
            ints = df["Int"].iloc[i]
            rbatt = df["RBAtt"].iloc[i]
            rbyds = df["RBYds"].iloc[i]
            rbtd = df["RBTD"].iloc[i]
            tgt = df["Tgt"].iloc[i]
            rec = df["Rec"].iloc[i]
            wryds = df["WRYds"].iloc[i]
            wrtd = df["WRTD"].iloc[i]
            fmb = df["Fmb"].iloc[i]
            fl = df["FL"].iloc[i]
            tottd = df["TOTTD"].iloc[i]
            twopm = df["2PM"].iloc[i]
            fantpt = df["FantPt"].iloc[i]
            ppr = df["PPRPt"].iloc[i]
            qbpts = (qbyds/25) + (qbtd*4) - (ints*2)
            rbpts = (rbyds/10) +(rbtd*6) - (fl*2)
            wrpts = (wryds/10) + (wrtd*6)
            wrppr = wrpts + rec
            if fantpt < 10:
                categories_3.append(0)
                categories_6.append(0)
            elif fantpt < 100:
                categories_3.append(1)
                categories_6.append(1)
            elif fantpt < 117:
                categories_3.append(1)
                categories_6.append(1)
            elif fantpt < 200:
                categories_3.append(2)
                categories_6.append(2)
            elif fantpt < 300:
                categories_3.append(2)
                categories_6.append(3)
            elif fantpt < 400:
                categories_3.append(2)
                categories_6.append(4)
            else:
                categories_3.append(2)
                categories_6.append(5)
            if qbatt == 0:
                qbypa.append(0)
            else:
                qbypa.append(round(qbyds/qbatt, 2))
            if not gp == 0:
                qbtd_pg.append(round(qbtd/gp, 2))
                cmp_pg.append(round(cmp/gp, 2))
                qbatt_pg.append(round(qbatt/gp, 2))
                qbyds_pg.append(round(qbyds/gp, 2))
                ints_pg.append(round(ints/gp, 2))
                rbatt_pg.append(round(rbatt/gp, 2))
                rbyds_pg.append(round(rbyds/gp, 2))
                rbtd_pg.append(round(rbtd/gp, 2))
                tgt_pg.append(round(tgt/gp, 2))
                rec_pg.append(round(rec/gp, 2))
                wryds_pg.append(round(wryds/gp, 2))
                wrtd_pg.append(round(wrtd/gp, 2))
                fmb_pg.append(round(fmb/gp, 2))
                tottd_pg.append(round(tottd/gp, 2))
                twopm_pg.append(round(twopm/gp, 2))
                fantpt_pg.append(round(fantpt/gp, 2))
                ppr_pg.append(round(ppr/gp, 2))
                qbppg.append(round(qbpts/gp, 2))
                rbppg.append(round(rbpts/gp, 2))
                wrppg.append(round(wrpts/gp, 2))
                wrppgppr.append(round(wrppr/gp, 2))
            else:
                qbtd_pg.append(0)
                cmp_pg.append(0)
                qbatt_pg.append(0)
                qbyds_pg.append(0)
                ints_pg.append(0)
                rbatt_pg.append(0)
                rbyds_pg.append(0)
                rbtd_pg.append(0)
                tgt_pg.append(0)
                rec_pg.append(0)
                wryds_pg.append(0)
                wrtd_pg.append(0)
                fmb_pg.append(0)
                tottd_pg.append(0)
                twopm_pg.append(0)
                fantpt_pg.append(0)
                ppr_pg.append(0)
                qbppg.append(0)
                rbppg.append(0)
                wrppg.append(0)
                wrppgppr.append(0)

        df.insert(8, "QBY/A", qbypa)
        df["QBTD/G"] = qbtd_pg
        df["Cmp/G"] = cmp_pg
        df["QBAtt/G"] = qbatt_pg
        df["QBYds/G"] = qbyds_pg
        df["Int/G"] = ints_pg
        df["RBAtt/G"] = rbatt_pg
        df["RBYds/G"] = rbyds_pg
        df["Rec/G"] = rec_pg
        df["WRYds/G"] = wryds_pg
        df["WRTD/G"] = wrtd_pg
        df["Fmb/G"] = fmb_pg
        df["TOTTD/G"] = tottd_pg
        df["2PM/G"] = twopm_pg
        df["FantPt/G"] = fantpt_pg
        df["PPR/G"] = ppr_pg
        df["QBPt/G"] = qbppg
        df["RBPt/G"] = rbppg
        df["WRPt/G"] = wrppg
        df["WRPPRPt/G"] = wrppgppr
        df["QB"] = qb
        df["RB"] = rb
        df["WR"] = wr
        df["TE"] = te
        df["3CATFANTPT"] = categories_3
        df["6CATFANTPT"] = categories_6
        #df["4CATFANTPT/G"] = categories2_4
        #df["8CATFANTPT/G"] = categories2_8
        
        df.to_csv(filename)
