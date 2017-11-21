const { exec } = require('child_process');
exec('python /home/kouohhashi/AIND-VUI-Capstone/pred2.py', (err, stdout, stderr) => {
  if (err) {
    // node couldn't execute the command
    console.log("err:", err)
    return;
  }

  // the *entire* stdout and stderr (buffered)
  console.log(`stdout: ${stdout}`);
  console.log(`stderr: ${stderr}`);
});
