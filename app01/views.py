from .forms import FileUploadForm
from django.shortcuts import render, redirect
from app01 import models
from MLfiles import CNN, LSTM, BiLSTM, Transfomer

global username, password, dataclass


# Create your views here.
def login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        print(username, password)
        queryset = models.UserInfo.objects.all()
        for obj in queryset:
            if obj.name == username and obj.password == password:
                return redirect('/upload/')
    return render(request, 'index.html')


def upload_file(request):
    global dataclass
    if request.method == 'POST':
        dataclass = request.POST.get('dataclass')
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('/analysis/')
    else:
        form = FileUploadForm()

    return render(request, 'upload.html', {'form': form})


def analysis(request):
    state = 0
    result1, acc1 = CNN.CNN(state, dataclass)
    result2, acc2 = LSTM.Lstm(state, dataclass)
    result3, acc3 = BiLSTM.BiLstm(state, dataclass)
    result4, acc4 = Transfomer.trans(state, dataclass)
    print(result1, acc1, result2, acc2, result3, acc3)
    return render(request, 'analysis.html',
                  {'result1': result1, 'result2': result2, 'result3': result3, 'result4': result4,
                   'acc1': acc1, 'acc2': acc2, 'acc3': acc3, 'acc4':acc4})
