const a = document.querySelector('#loginA');
const div = document.querySelector('.title');
const close = document.querySelector('#close');
const submit =  document.querySelector('#submit');
const bg = document.querySelector('.bg');
const login = document.querySelector('.login');
// 弹窗的关闭和出现
a.onclick=function(){
    bg.style.display = 'block';
    login.style.display='block';
}
close.onclick=function(){
    bg.style.display = 'none';
    login.style.display='none';
}

submit.addEventListener("click", getValues);
submit.addEventListener("click", close_);
function close_(){
    bg.style.display = 'none';
    login.style.display='none';
}
//设置让其移动
div.addEventListener('mousedown',function(e){
    const x = e.pageX - login.offsetLeft;
    const y = e.pageY - login.offsetTop;
    document.addEventListener("mousemove",move)
    function move(e){
        const newx = e.pageX - x;
        const newy = e.pageY - y;
        login.style.left=newx+'px';
    login.style.top=newy+'px';
    }
    document.addEventListener('mouseup',function(e){
        document.removeEventListener("mousemove",move);
    })
    })

function getValues() {
    const username = document.getElementById("username").value;
    const password = document.getElementById("password").value;
    alert("用户名：" + username +"\n " + "密 码：" + password);
}
