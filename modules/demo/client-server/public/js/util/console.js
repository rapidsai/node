var td             = document.getElementById('textdiv');
var cw             = document.getElementById('console-window');
var displayCompute = true;  // default value of console-window display is true

// Update console div with the msg received
function updateConsole(msg) {
  if (displayCompute) {
    td.innerHTML += '<br> _> ' + msg + ' <br>';
    td.scrollTop = td.scrollHeight;
  }
}

// toggle the console div on click of `toggle-console` button
document.getElementById('toggle-console').onclick = () => {
  if (displayCompute == true) {
    displayCompute   = false;
    cw.style.display = 'none';
  } else {
    displayCompute     = true;
    cw.style.display   = 'block';
    td.style.overflowY = 'auto';
    td.style.height    = '80%';
  }
};
