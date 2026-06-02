document.addEventListener('DOMContentLoaded', function() {
    // --- 0. Sidebar Active State Logic ---
    const navLinks = document.querySelectorAll('.nav-links li');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            navLinks.forEach(l => l.classList.remove('active'));
            this.classList.add('active');
        });
    });

    // --- 1. Chart.js Line Chart (Actual vs Prediction 1-Bulan) ---
    const ctx = document.getElementById('inflationChart').getContext('2d');
    
    // Gradien untuk area bawah garis
    const gradientActual = ctx.createLinearGradient(0, 0, 0, 300);
    gradientActual.addColorStop(0, 'rgba(160, 160, 165, 0.2)');
    gradientActual.addColorStop(1, 'rgba(160, 160, 165, 0)');
    
    const gradientPred = ctx.createLinearGradient(0, 0, 0, 300);
    gradientPred.addColorStop(0, 'rgba(188, 248, 70, 0.4)');
    gradientPred.addColorStop(1, 'rgba(188, 248, 70, 0)');

    // Karena forecast hanya 1 bulan ke depan (Maret 2026),
    // kita tampilkan data aktual 6 bulan terakhir agar grafiknya terlihat informatif.
    const labels = ['Okt 2025', 'Nov 2025', 'Des 2025', 'Jan 2026', 'Feb 2026', 'Mar 2026 (Prediksi)'];
    
    // Data Aktual: Ada data dari Okt 2025 sampai Feb 2026, lalu di Mar 2026 kosong (null)
    const dataActual = [0.17, 0.38, 0.41, 0.04, 0.21, null];
    
    // Data Prediksi: Agar garis terhubung mulus, kita mulai dari titik aktual terakhir (Feb 2026) menuju prediksi (Mar 2026)
    const dataPred = [null, null, null, null, 0.21, 2.14];

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Inflasi Aktual',
                    data: dataActual,
                    borderColor: '#a0a0a5',
                    backgroundColor: gradientActual,
                    borderWidth: 2,
                    tension: 0.1,
                    fill: true,
                    pointRadius: 4,
                    pointHoverRadius: 6
                },
                {
                    label: 'Prediksi LSTM (1 Bulan)',
                    data: dataPred,
                    borderColor: '#bcf846',
                    backgroundColor: gradientPred,
                    borderWidth: 3,
                    borderDash: [5, 5],
                    tension: 0.1,
                    fill: true,
                    pointBackgroundColor: '#121212',
                    pointBorderColor: '#bcf846',
                    pointBorderWidth: 2,
                    pointRadius: 6,
                    pointHoverRadius: 8
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    align: 'end',
                    labels: {
                        color: '#a0a0a5',
                        usePointStyle: true,
                        boxWidth: 8
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(37, 38, 44, 0.9)',
                    titleColor: '#fff',
                    bodyColor: '#bcf846',
                    borderColor: 'rgba(188, 248, 70, 0.3)',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: false
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#a0a0a5'
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#a0a0a5',
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            },
            interaction: {
                mode: 'index',
                intersect: false,
            }
        }
    });

    // 2. Interactive Simulator
    const slider = document.getElementById('inflasiSlider');
    const inflasiVal = document.getElementById('inflasiVal');
    const dayaBeliResult = document.getElementById('dayaBeliResult');
    
    // Baseline pengeluaran per kapita (misal 1,450,000)
    const basePengeluaran = 1450000;
    
    // Format mata uang Rupiah
    const formatIDR = (number) => {
        return new Intl.NumberFormat('id-ID', {
            style: 'currency',
            currency: 'IDR',
            minimumFractionDigits: 0,
            maximumFractionDigits: 0
        }).format(number);
    };

    slider.addEventListener('input', function(e) {
        const val = parseFloat(e.target.value);
        inflasiVal.textContent = val.toFixed(1) + '%';
        
        // Simulasi Regresi Sederhana: 
        // Setiap kenaikan 1% inflasi menurunkan daya beli sebesar 15,000 Rupiah
        const koefisien = 15000;
        const penurunan = val * koefisien;
        let estimasi = basePengeluaran - penurunan;
        
        dayaBeliResult.textContent = formatIDR(estimasi);
        
        // Tambahkan efek visual (mengubah warna jika terlalu rendah)
        if (estimasi < 1350000) {
            dayaBeliResult.style.color = '#ff5252';
        } else {
            dayaBeliResult.style.color = '#bcf846';
        }
    });
});
