from flask import *

from src.rf import *

app=Flask(__name__)




@app.route("/")
def firtpage():
    return render_template("first page.html")

@app.route("/hhhh",methods=['post'])
def hhhh():
    lis=[]
    gender=request.form['radiobutton']
    age=request.form['textfield']
    b_dgtl=request.form['select1']
    a_dgtl = request.form['select2']
    b_dgtm=request.form['textfield4']
    a_dgtm = request.form['textfield5']
    b_usedgsdy=request.form['select3']
    a_usedgsdy = request.form['select4']
    b_distract= request.form['select5']
    a_distract = request.form['select6']
    b_slptm= request.form['select7']
    a_slptm = request.form['select8']
    b_afctslp=request.form['select9']
    a_afctslp=request.form['select10']
    b_tir=request.form['select11']
    a_tir=request.form['select12']
    on_sd= request.form['select13']
    isolatn= request.form['select14']
    unlrn_btr = request.form['select15']
    lazy= request.form['select16']
    nervs_tnsn= request.form['select17']
    psy_keyfctr= request.form['select18']
    tlsbuy_frus= request.form['select19']
    dnt_rcmnd = request.form['select20']
    lock_dprn= request.form['select21']
    low_acadmcs= request.form['select22']
    asgmtsub_confn = request.form['select23']
    ftof_btr = request.form['select24']
    exms_nrvs= request.form['select25']

    lis.append(gender)
    lis.append(age)
    lis.append(b_dgtl)
    lis.append(a_dgtl)
    lis.append(b_dgtm)
    lis.append(a_dgtm)
    lis.append(b_usedgsdy)
    lis.append(a_usedgsdy)
    lis.append(b_distract)
    lis.append(a_distract)
    lis.append(b_slptm)
    lis.append(a_slptm)
    lis.append(b_afctslp)
    lis.append(a_afctslp)
    lis.append(b_tir)
    lis.append(a_tir)
    lis.append(on_sd)
    lis.append(isolatn)
    lis.append(unlrn_btr)
    lis.append(lazy)
    lis.append(nervs_tnsn)
    lis.append(psy_keyfctr)
    lis.append(tlsbuy_frus)
    lis.append(dnt_rcmnd)
    lis.append(lock_dprn)
    lis.append(low_acadmcs)
    lis.append(asgmtsub_confn)
    lis.append(ftof_btr)
    lis.append(exms_nrvs)



    out=predictfn(lis)
    print(out)

    return out


app.run(debug=True)