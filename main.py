import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from openpyxl import styles, load_workbook
import os
import jinja2
import pdfkit

# `Nan` value removal for DataFrame
def remnan(df, fill='--'):
    indexdict = {}
    for x in df.values:
        for i,x1 in enumerate(x,0):
            if i not in indexdict.keys():
                if x1 != fill:
                    indexdict[i] = [round(x1,4)]
            else:
                if x1 != fill:
                    indexdict[i].append(round(x1,4))
    order_list = list(indexdict.keys())
    max1 = max(len(x) for x in indexdict.values())
    order_list.sort()
    indexdict2= {}
    for num2 in order_list:
        newlist = indexdict[num2]
        length = len(newlist)
        if length != max1:
            cor = max1 -length
            newlist.extend(cor*[np.nan])
        indexdict2[num2] = newlist
    dt = pd.DataFrame(columns=df.columns)
    for cols,ent in zip(df.columns,indexdict2.values()):
        dt.loc[:, cols] =ent
    dt.index = range(1,len(dt)+1)
    return dt

# rounds to 4 decimal places
def round4(num):
    string = str(round(num,4))
    cut = string.index('.')
    first_string = string[0:cut]
    sub_string = string[cut:]
    lent = len(sub_string)
    if lent < 5:
        outfloat = first_string+sub_string+'0'*(5-lent)
    else:
        outfloat = first_string+sub_string
    return outfloat

class reportcreate:

    def __init__(self, header='Main Title', blocks={}):
        pass
        

def anovarun(file='data.csv', order=['supp','dose','len']):
    """
    Docstring: Function generates runs an Anova 2 ways analysis from a csv file

    """
    # Convert csv data to `pandas.DataFrame` [df2]
    ddf2 = pd.read_csv(file, header=0, sep=',')
    if order:
        emplist = []
        for mat in order:
            emplist.append(mat in ddf2.columns)
        if sum(emplist) == 3:
            ddf2 = ddf2[order]

    fcols = list(ddf2.columns)[0:2] # factor columns for DataFrame
    vcols = list(ddf2.columns)[2] # value columns for DataFrame
    ddf2.sort_values(by=[fcols[0],fcols[1]], inplace=True)
    ddf2.index = range(1, len(ddf2)+1) # Renumber [df2]


    # Pivot table based on the 1st and 2nd factors
    df2_pivot = ddf2.pivot_table(columns=fcols, values=vcols, index=ddf2.index, fill_value='--')

    df2_pivot = remnan(df2_pivot) # Removes nan values from [df2_pivot]
    print(df2_pivot)
    dm = df2_pivot.melt(value_name=vcols) # Melt table back to original [dm]
    dmgr = dm.groupby(fcols).mean() # Grouping Melted table and calculating mean [dmgr]
    dmgr_unstack = dmgr.unstack(1) # unstacking data of 1st factor on index

    # Calculation of mean values
    dmgr_unstack_mean = dmgr_unstack.copy() # DataFrame copy to [dmgr_unstack_mean]
    dmgr_unstack_mean.loc[:,(vcols,'Average')] = dmgr_unstack_mean.apply(lambda x: round(np.mean(x),1), axis=1) # calculating mean of values along row as [['Average']]
    dmgr_unstack_mean.loc['Average',:] = dmgr_unstack_mean.apply(np.mean, axis=0) # calculating mean of values along column on index [['Average']]

    # Sum of Square of 1st factor
    def catall(df):
        li5=[]
        li7=[]
        for nums1,li6 in zip(range(1,len(df.columns)+1),df.columns):
            if nums1==1:
                li5.append(li6[0])
            elif nums1==(len(df2_pivot.columns)/2)+1:
                li5.append(li6[0])
        for v in df.iteritems():
            li7.append(v[1].count())
        return li7,li5
    nv,lab = catall(df2_pivot) # finding number of data for 1st factors [nv]
    mv = dmgr_unstack_mean.loc[:,(vcols,'Average')].values[0:2] # Mean of first category on first factor
    xv = dmgr_unstack_mean.loc['Average',(vcols,'Average')] # Data overall mean
    chk = int(len(nv)/len(mv))
    index = 0
    ss1v = 0
    emptylist =[]
    for me_in in mv: # Mathematical calculation for [ss1v]
        for m2 in range(index, chk+index):
            emptylist.append((nv[m2],me_in))
            ss1v = ss1v + (nv[m2]*(me_in-xv)**2)
        index = m2+1
    ss1v = round(ss1v,4)

    # Sum of Square of 2nd factor
    nps = dmgr_unstack_mean.iloc[-1,:].values[0:3]
    ss2v = 0
    ## Loop to solve [ss2v]
    for cc in lab:
        ins = df2_pivot.loc[:,(cc)].count().values
        for no,nums in zip(ins,nps):
            ss2v = ss2v + no*(nums - xv)**2
    ss2v = round(ss2v,4)

    # Sum of Square within (Error)
    df2_pivot_ssw = df2_pivot.copy() # Copy [df2_pivot] to [df2_pivot_ssw]
    
    ## function to perform iteration on apply method [df2_pivot_ssw]
    def sswdo(arg,x):
        li1 = []
        arg = arg.values
        for num2,xi in zip(arg,x):
            if str(num2)!='--':
                li1.append((num2-xi)**2)
        out =sum(li1)
        return out
    ref = len(dmgr_unstack_mean.columns)-1
    li2 = []
    for seg in range(len(lab)):
        li2.extend(list(dmgr_unstack_mean.iloc[seg,0:ref].values))
    df2_pivot_ssw.fillna('--', inplace=True)
    ssw = round(sum(df2_pivot_ssw.apply(sswdo, x=li2, axis=1)),4)

    # Sum of Square Total
    sst = round(dm.iloc[:,-1].apply(lambda x: (x-xv)**2 if x>1 else None).sum(),2) # (x-xv)**2: for sum of square total [sst]

    # Sum of square for both factors [ssb]
    ssb = round(sst-ss1v-ss2v-ssw,2)

    # Degree of Freedom calcution
    df1 = len(dmgr_unstack_mean.index)-2 # degree of freedom for factor 1
    df2 = len(dmgr_unstack_mean.columns)-2 # degree of freedom for factor 2
    ref = len(dmgr_unstack_mean.columns)-1
    out = 0
    for num3 in nv:
        out = out + num3-1
    dfw = out # degree of freedom for within
    
    dfb = df1 *df2 # degree of freedom for both
    dft = df1+df2+dfw+dfb # degree of freedon both

    # Create Anova table
    anova = pd.DataFrame(columns=['SS','DF','MEAN_SUM','F'])
    anova.index.name = 'SOURCES'
    
    # Fill Anova table
    row1 = df2_pivot.columns.names[0].upper()
    row2 = df2_pivot.columns.names[1].upper()
    row3 = df2_pivot.columns.names[0].upper()+':'+df2_pivot.columns.names[1].upper()

    anova.loc[row1,:] = [round4(ss1v),df1,round4(round(ss1v/df1,4)),round4(round((ss1v/df1)/(ssw/dfw),4))]
    anova.loc[row2,:] = [round4(ss2v),df2,round4(round(ss2v/df2,4)),round4(round((ss2v/df2)/(ssw/dfw),4))]
    anova.loc[row3,:] = [round4(ssb),dfb,round4(round(ssb/dfb,4)),round4(round((ssb/dfb)/(ssw/dfw),4))]
    anova.loc['RESIDUAL'.upper(),:] = [round4(ssw),dfw,round4(round(ssw/dfw,4)),'']
    anova.loc['TOTAL'.upper(),:] = [round4(sst),dft,'','']

    # Profile plot preparation
    li4 = []
    for nums1,li3 in zip(range(1,len(df2_pivot.columns)+1),df2_pivot.columns):
        if nums1==1:
            li4.append(li3[0])
        elif nums1==(len(df2_pivot.columns)/2)+1:
            li4.append(li3[0])
    fig0 = plt.figure(2,figsize=(10,10))
    ax = plt.subplot(111, label='Axes 1')
    dmgr_unstack_plt = dmgr.unstack(0)
    dmgr_unstack_plt.plot(ax=ax, marker='H')
    plt.legend(li4)
    plt.tight_layout()
    ax.set_ylabel(ylabel='Estimated Marginal means', fontdict={'size':20})
    ax.set_xlabel(xlabel=ax.get_xlabel(), fontdict={'size':20})
    fig0.savefig('plot.png',format='png', bbox_inches='tight')
    plt.show()
    # change NaN values to '--'
    df2_pivot.fillna('--', inplace=True)

    # Report creation into Pdf
    filename = "ereport.xlsx"
    anova.to_excel(filename, sheet_name='Table')
    wb = load_workbook(filename)
    ws = wb.worksheets[0]

    # Setting up border styles
    bleftv = styles.Side(style='thin')
    brightv = styles.Side(style='thin')
    btopv = styles.Side(style='thin')
    bbottomv = styles.Side(style='thin')
    borderv = styles.borders.Border(bleftv,brightv,btopv,bbottomv)
    for cols_cells in ws.columns:
        length = max(len(str(cell.value)) for cell in cols_cells) + 5
        ws.column_dimensions[cols_cells[0].column_letter].width = length
        for cell in cols_cells:
            cell.border = borderv
    wb.save(filename)
    wb.close()
    
    # Generating report to html
    reportname = 'report'
    def spacefact(df):
        max1 = 0
        for idf in df.iterrows():
            n_len = len(str(idf[0]))
            if n_len > max1:
                max1 = n_len
        return max1
    templateLoader = jinja2.FileSystemLoader(searchpath="./")
    templateEnv = jinja2.Environment(loader=templateLoader, extensions=['jinja2.ext.do'])
    TEMPLATE_FILE = "template.jinja"
    template = templateEnv.get_template(TEMPLATE_FILE)
    outputText = template.render(ddf2=ddf2,df=df2_pivot, df2=dmgr_unstack_mean, ano=anova,zip=zip, list=list, len=len, space=spacefact)
    html_file = open(f'{reportname}.html', 'w')
    html_file.write(outputText)
    html_file.close()
    
    # Converting Html report to pdf
    path_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
    config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
    pdfkit.from_file(f'{reportname}.html', f'{reportname}.pdf', configuration=config)

    os.system(f'cmd /c "explorer {reportname}.pdf"')

if __name__ == "__main__":
    anovarun()