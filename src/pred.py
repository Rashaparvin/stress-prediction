from flask import *

from src.rf import *
app=Flask(__name__)




@app.route("/")
def firtpage():
    return render_template("first page.html")

@app.route("/sp",methods=['post'])
def sp():
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

    lis.append(int(gender))
    lis.append(int(age))
    lis.append(int(b_dgtl))
    lis.append(int(a_dgtl))
    lis.append(int(b_dgtm))
    lis.append(int(a_dgtm))
    lis.append(int(b_usedgsdy))
    lis.append(int(a_usedgsdy))
    lis.append(int(b_distract))
    lis.append(int(a_distract))
    lis.append(int(b_slptm))
    lis.append(int(a_slptm))
    lis.append(int(b_afctslp))
    lis.append(int(a_afctslp))
    lis.append(int(b_tir))
    lis.append(int(a_tir))
    lis.append(int(on_sd))
    lis.append(int(isolatn))
    lis.append(int(unlrn_btr))
    lis.append(int(lazy))
    lis.append(int(nervs_tnsn))
    lis.append(int(psy_keyfctr))
    lis.append(int(tlsbuy_frus))
    lis.append(int(dnt_rcmnd))
    lis.append(int(lock_dprn))
    lis.append(int(low_acadmcs))
    lis.append(int(asgmtsub_confn))
    lis.append(int(ftof_btr))
    lis.append(int(exms_nrvs))
    #
    # print("=====================================+++++++++++++")
    # print("=====================================+++++++++++++")
    # print("=====================================+++++++++++++")
    # print(lis)
    # print("++++++++++++++++++++++++++++++++++++++============")
    # print("++++++++++++++++++++++++++++++++++++++============")
    # print("++++++++++++++++++++++++++++++++++++++============")
    out=str(predictfn(lis))

    print(out,"===========",type(out))

    return render_template("stressview.html",val=out)


app.run(debug=True)