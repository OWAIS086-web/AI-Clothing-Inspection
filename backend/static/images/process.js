// Initialize SignalR connection
const connection = new signalR.HubConnectionBuilder()
    .withUrl("/InspectionHub")
    .withAutomaticReconnect()
    .configureLogging(signalR.LogLevel.Information)
    .build();

connection.start().then(function () {
    console.log("Connected!");
}).catch(function (err) {
    console.error(err.toString());
});

connection.serverTimeoutInMilliseconds = 60000; // 60 seconds
connection.keepAliveIntervalInMilliseconds = 15000; // 15 seconds

// Handle refreshpreprocess event
connection.on("refreshpreprocess", function (data) {
    console.log(data);
    $("#logs").text("");
    $("#defectimgsdiv").html("");
    $(".loaderDiv").addClass("d-none");

    // Process and display images
    let itm = '';
    for (let i = 0; i < data.pdf.length; i++) {
        let imgData = data.pdf[i];
        if (imgData) {
            itm += `
                <div class="owl-item">
                    <img class="img-fluid" src="data:image/png;base64,${imgData}" alt="Alternate Text">
                </div>`;
        } else {
            console.error('Invalid image data at index ' + i);
        }
    }

    $("#listimg").html(itm);

    $('#deflist').owlCarousel({
        loop: true,
        margin: 10,
        responsiveClass: true,
        responsive: {
            0: {
                items: 1,
                nav: true
            },
            600: {
                items: 1,
                nav: false
            },
            1000: {
                items: 1,
                nav: true,
                loop: false
            }
        }
    });
    $('#deflist').trigger('refresh.owl.carousel');
    $(".pages").removeClass('d-none');

    // Process and display fiber content
    if (data.fiber_content !== null) {
        const table = document.createElement('table');
        table.classList.add('table', 'table-inspection', 'table-bordered', 'mb-3');
        const maxRowLength = Math.max(...data.fiber_content.map(column => column.length));

        for (let row = 0; row < maxRowLength; row++) {
            const tr = document.createElement('tr');

            data.fiber_content.forEach(column => {
                const td = document.createElement('td');
                if (row < column.length) {
                    td.textContent = column[row];
                }
                tr.appendChild(td);
            });

            table.appendChild(tr);
        }

        document.getElementById('tblcontainer').appendChild(table);
    }

    // Process and display care instructions
    if (data.care_instr !== null) {
        const table = document.createElement('table');
        table.classList.add('table', 'table-inspection', 'table-bordered', 'mb-3');

        data.care_instr.forEach(s => {
            const tr = document.createElement('tr');
            const th2 = document.createElement('th');
            th2.textContent = s;
            tr.appendChild(th2);
            table.appendChild(tr);
        });

        document.getElementById('care_instr_container').appendChild(table);
    }

    // Process and display extra care instructions
    if (data.excare_instr !== null) {
        const table = document.createElement('table');
        table.classList.add('table', 'table-inspection', 'table-bordered', 'mb-3');

        data.excare_instr.forEach(s => {
            const tr = document.createElement('tr');
            const th2 = document.createElement('th');
            th2.textContent = s;
            tr.appendChild(th2);
            table.appendChild(tr);
        });

        document.getElementById('excare_instr_cont').appendChild(table);
    }

    // Process and display symbols
    if (data.symbols !== null) {
        const table = document.createElement('table');
        table.classList.add('table', 'table-inspection', 'table-bordered', 'mb-3');
        const maxRowLength = Math.max(...data.symbols.map(column => column.length));

        for (let row = 0; row < maxRowLength; row++) {
            const tr = document.createElement('tr');

            data.symbols.forEach((column, columnIndex) => {
                const td = document.createElement('td');
                if (row < column.length) {
                    if (columnIndex === 2) {
                        const img = document.createElement('img');
                        img.src = "data:image/png;base64," + column[row];
                        img.alt = 'Symbol Image';
                        img.classList.add('img-fluid');
                        td.appendChild(img);
                    } else {
                        td.textContent = column[row];
                    }
                }
                tr.appendChild(td);
            });

            table.appendChild(tr);
        }

        document.getElementById('symbols').appendChild(table);
    }
});

// Handle finalresult event
connection.on("finalresult", function (data) {
    $(".loaderDiv").addClass("d-none");
    const decodedText = atob(data.logs);
    $("#time").text(data.time_taken);

    $("#listimg").html('');
    const logsDiv = document.getElementById('logs');
    logsDiv.innerHTML = decodedText.replace(/\n/g, '<br>');
    
    // Process and display final images
    let itm = '';
    for (let i = 0; i < data.pdf_image.length; i++) {
        let imgData = data.pdf_image[i];
        if (imgData) {
            itm += `
                <div class="owl-item">
                    <img class="img-fluid" src="data:image/png;base64,${imgData}" alt="Alternate Text">
                </div>`;
        } else {
            console.error('Invalid image data at index ' + i);
        }
    }

    $("#listimg").html(itm);

    $('#deflist').owlCarousel({
        loop: true,
        margin: 10,
        dots: false,
        responsiveClass: true,
        onInitialized: counter,
        onChanged: counter,
        responsive: {
            0: {
                items: 1,
                nav: true
            },
            600: {
                items: 1,
                nav: false
            },
            1000: {
                items: 1,
                nav: true,
                loop: false
            }
        }
    });
    $('#deflist').trigger('refresh.owl.carousel');
    $(".pages").removeClass('d-none');

    // Process and display defect list images
    let itm2 = '';
    for (let i = 0; i < data.defect_list.length; i++) {
        let imgData = data.defect_list[i];
        if (imgData) {
            itm2 += `
                <div class="owl-item">
                    <img class="img-fluid" src="data:image/png;base64,${imgData}" alt="Alternate Text">
                </div>`;
        } else {
            console.error('Invalid image data at index ' + i);
        }
    }

    $("#listimg2").html(itm2);

    $('#deflist2').owlCarousel({
        loop: true,
        margin: 10,
        dots: false,
        responsiveClass: true,
        onInitialized: counter,
        onChanged: counter,
        responsive: {
            0: {
                items: 1,
                nav: true
            },
            600: {
                items: 1,
                nav: false
            },
            1000: {
                items: 1,
                nav: true,
                loop: false
            }
        }
    });
    $('#deflist2').trigger('refresh.owl.carousel');
});
