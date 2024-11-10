const fs = require('fs');
const QuickChart = require('quickchart-js');
const path = require('path');


const resultsFile = path.join(__dirname, 'benchmark_results.json');
let results = {};


if (fs.existsSync(resultsFile)) {
    results = JSON.parse(fs.readFileSync(resultsFile, 'utf8'));
} else {
    console.error('benchmark_results.json not found.');
    process.exit(1);
}


let data = Object.keys(results).map(pattern => ({
    statesLength: results[pattern]['states_length'],
    averageTime: results[pattern]['average_time'],
    outlinesTime: results[pattern]['outlines_time'] || 0 
}));

data.sort((a, b) => a.statesLength - b.statesLength);


const stateLengths = data.map(item => item.statesLength);  
const times = data.map(item => item.averageTime);         
const outlinesTimes = data.map(item => item.outlinesTime); 


const datasets = [
    {
        label: 'Faster-Outlines',
        data: times,
        fill: false,
        borderColor: 'rgb(255, 99, 132)',  
        tension: 0.1
    },
    {
        label: 'Outlines',
        data: outlinesTimes,
        fill: false,
        borderColor: 'rgb(54, 162, 235)',  
        tension: 0.1
    },
    {
        label: 'Time Inference Is Blocked (avg)',
        data: Array(stateLengths.length).fill(0.02),  
        fill: false,
        borderColor: 'rgb(255, 206, 86)',  
        borderDash: [5, 5],  
        tension: 0.1
    }
];


const chart = new QuickChart();
chart.setConfig({
    type: 'line',
    data: {
        labels: stateLengths,
        datasets: datasets     
    },
    options: {
        responsive: true,
        plugins: {
            title: {
                display: true,
                text: 'Regex Compilation Time vs. State Length',
                color: '#ffffff' 
            },
            legend: {
                labels: {
                    color: '#ffffff' 
                }
            }
        },
        scales: {
            x: {
                title: {
                    display: true,
                    text: 'Regex State Length',
                    color: '#ffffff'  
                },
                ticks: {
                    color: '#ffffff'  
                },
                type: 'logarithmic'
            },
            y: {
                title: {
                    display: true,
                    text: 'Average Compilation Time (s)',
                    color: '#ffffff' 
                },
                ticks: {
                    color: '#ffffff'  
                },
                type: 'logarithmic'
            }
        }
    }
}).setBackgroundColor('#2c2c2c')  
  .setWidth(500)
  .setHeight(400);


chart.toFile('assets/benchmark.png')
    .then(() => {
        console.log('Graph generated and saved as assets/benchmark.png');
    })
    .catch((error) => {
        console.error('Error generating graph:', error);
    });

function getRandomColor() {
    const r = Math.floor(Math.random() * 255);
    const g = Math.floor(Math.random() * 255);
    const b = Math.floor(Math.random() * 255);
    return `rgb(${r},${g},${b})`;
}
