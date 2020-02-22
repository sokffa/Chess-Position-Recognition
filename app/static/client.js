var el = x => document.getElementById(x);

function showPicker() {
  el("file-input").click();
}

function showPicked(input) {
  el("upload-label").innerHTML = input.files[0].name;
  var reader = new FileReader();
  reader.onload = function(e) {
    el("image-picked").src = e.target.result;
    el("image-picked").className = "";
  };
  reader.readAsDataURL(input.files[0]);
}

function analyze() {
  var uploadFiles = el("file-input").files;
  var board_orientation = document.getElementsByName("board_orientation");
  for (var i = 0, length = board_orientation.length; i < length; i++) {
  if (board_orientation[i].checked) {
    board_orientation = board_orientation[i].value;
    break;
  }
  }
  var movesNext = document.getElementsByName("movesNext");
  for (var i = 0, length = movesNext.length; i < length; i++) {
  if (movesNext[i].checked) {
    movesNext = movesNext[i].value;
    break;
  }}

  if (uploadFiles.length !== 1) alert("Please select a file to analyze!");

  el("analyze-button").innerHTML = "Analyzing...";
  var xhr = new XMLHttpRequest();
  var loc = window.location;
  xhr.open("POST", `${loc.protocol}//${loc.hostname}:${loc.port}/analyze`,
    true);
  xhr.onerror = function() {
    alert(xhr.responseText);
  };
  xhr.onload = function(e) {
    if (this.readyState === 4) {
      var response = JSON.parse(e.target.responseText);
      el("result-label").innerHTML = `Result = ${response["result"]}`;
    }
    el("analyze-button").innerHTML = "Analyze";
  };

  var fileData = new FormData();
  fileData.append("file", uploadFiles[0]);
  fileData.append("board_orientation" , board_orientation);
  fileData.append("movesNext" , movesNext);
  xhr.send(fileData);
}

