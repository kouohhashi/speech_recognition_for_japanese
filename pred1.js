// var PythonShell = require('python-shell');
//
// var options = {
//   pythonPath: '/home/kouohhashi/anaconda3/bin/python',
//   mode: 'text',
//   args: [540, -6, 67, 1800, 18]
// };
// var pyshell = new PythonShell("/home/kouohhashi/AIND-VUI-Capstone/pred1.py", options);
//
// pyshell.on('message', function (message) {
//     // received a message sent from the Python script (a simple "print" statement)
//     console.log("message 1:");
//     console.log(message);
// });
//
// // end the input stream and allow the process to exit
// pyshell.end(function (err) {
//     if (err){
//         console.log(err);
//     };
//     console.log('\n');
//     console.log('finished');
// });
//

var PythonShell = require('python-shell');

var options = {
  mode: 'text',
  pythonPath: '/home/kouohhashi/anaconda3/bin/python',
  pythonOptions: ['-u'],
  scriptPath: '/home/kouohhashi/AIND-VUI-Capstone',
  args: ['value1', 'value2', 'value3']
};

PythonShell.run('pred1.py', options, function (err, results) {
  if (err) {
    console.log(err);
    return;
  }
  // results is an array consisting of messages collected during execution
  console.log('results:', results);
});
