async function fetchMetrics() {
    const response = await fetch('/server');
    const data = await response.json();
    
    document.getElementById('cpuUsage').innerText = data.cpu_usage + '%';
    document.getElementById('memoryUsage').innerText = `Total: ${formatBytes(data.memory_total)}, Used: ${formatBytes(data.memory_used)}`;
    document.getElementById('storageUsage').innerText = `Total: ${formatBytes(data.storage_total)}, Used: ${formatBytes(data.storage_used)}`;
    document.getElementById('networkUsage').innerText = `Sent: ${formatBytes(data.net_sent)}, Received: ${formatBytes(data.net_recv)}`;
    document.getElementById('uptime').innerText = data.uptime;
    document.getElementById('loadAvg').innerText = `1m: ${data.load_avg[0]}, 5m: ${data.load_avg[1]}, 15m: ${data.load_avg[2]}`;
    document.getElementById('openFiles').innerText = data.open_files;
    document.getElementById('connections').innerText = data.connections;
    document.getElementById('systemInfo').innerText = `Hostname: ${data.hostname}, Platform: ${data.platform}, Release: ${data.platform_release}, Version: ${data.platform_version}`;
    

}

function formatBytes(bytes) {
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    if (bytes == 0) return '0 Byte';
    const i = parseInt(Math.floor(Math.log(bytes) / Math.log(1024)));
    return Math.round(bytes / Math.pow(1024, i), 2) + ' ' + sizes[i];
}

setInterval(fetchMetrics, 5000); // Refresh every 5 seconds
fetchMetrics();