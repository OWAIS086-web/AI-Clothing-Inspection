const socket = io();

socket.on('preprocessed_results', function(data) {
    document.getElementById('loader').style.display = 'none';
    if (data.keyword) {
        createTable('keyword_content_table', 'Query Keywords', data.keyword);
    }
    if (data.fiber_content) {
        createTable('fiber_content_table', 'Fiber Content', data.fiber_content);
    }
    if (data.care_instr) {
        createList('care_instr_table', 'Care Instructions', data.care_instr);
    }
    if (data.excare_instr) {
        createList('excare_instr_table', 'Extra Care Instructions', data.excare_instr);
    }
    if (data.symbols) {
        createSymbolsTable('symbols_table', 'Symbols', data.symbols);
    }
    if (data.pdf) {
        displayPngImages('pdf_display', data.pdf);
    }
    if (data.fit_name) {
        displayList('fit_name_display', 'FIT NAME', data.fit_name);
    }
    if (data.clr) {
        displayList('clr_display', 'CLR #', data.clr);
    }
    if (data.size) {
        displayList('size_display', 'SIZE', data.size);
    }
    
});

function createTable(elementId, heading, tableData) {
    let table = document.createElement('table');
    table.className = 'table';
    let headerRow = document.createElement('tr');

    tableData[0].forEach(header => {
        let th = document.createElement('th');
        th.textContent = header;
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);

    tableData.slice(1).forEach(rowData => {
        let row = document.createElement('tr');
        rowData.forEach(cellData => {
            let td = document.createElement('td');
            td.textContent = cellData;
            row.appendChild(td);
        });
        table.appendChild(row);
    });

    let container = document.getElementById(elementId);
    container.innerHTML = `<h3>${heading}</h3>`;
    container.appendChild(table);
}

function createList(elementId, heading, listData) {
    let ul = document.createElement('ul');
    ul.className = 'list';

    listData.forEach(item => {
        let li = document.createElement('li');
        li.textContent = item;
        ul.appendChild(li);
    });

    let container = document.getElementById(elementId);
    container.innerHTML = `<h3>${heading}</h3>`;
    container.appendChild(ul);
}

function createSymbolsTable(elementId, heading, symbolsData) {
    const table = document.createElement('table');
    table.classList.add('table', 'table-inspection', 'table-bordered', 'mb-3');
    const maxRowLength = Math.max(...symbolsData.map(column => column.length));

    for (let row = 0; row < maxRowLength; row++) {
        const tr = document.createElement('tr');

        symbolsData.forEach((column, columnIndex) => {
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

    const container = document.getElementById(elementId);
    container.innerHTML = `<h3>${heading}</h3>`;
    container.appendChild(table);
}

function displayPngImages(elementId, pngData) {
    let container = document.getElementById(elementId);
    container.innerHTML = '';

    // Create a div with class "style-153"
    const styleDiv = document.createElement('div');
    styleDiv.classList.add('style-153');

    // Create a Swiper container
    const swiperContainer = document.createElement('div');
    swiperContainer.classList.add('swiper-container');

    // Create a Swiper wrapper
    const swiperWrapper = document.createElement('div');
    swiperWrapper.classList.add('swiper-wrapper');

    pngData.forEach((pngBase64, index) => {
        // Create a slide for each image
        const slide = document.createElement('div');
        slide.classList.add('swiper-slide');

        // Create an image element and set the source
        const imgElement = document.createElement('img');
        imgElement.src = 'data:image/png;base64,' + pngBase64;
        imgElement.alt = `PNG Image ${index + 1}`;
        imgElement.classList.add('img-fluid');

        // Append image to the slide
        slide.appendChild(imgElement);
        swiperWrapper.appendChild(slide);
    });

    // Append Swiper wrapper to the Swiper container
    swiperContainer.appendChild(swiperWrapper);

    // Append Swiper container to the style-153 div
    styleDiv.appendChild(swiperContainer);

    // Append style-153 div to the main container
    container.appendChild(styleDiv);

    // Initialize Swiper with configuration for single-page display
    new Swiper('.swiper-container', {
        loop: true,
        slidesPerView: 1,
        spaceBetween: 0,
        pagination: {
            el: '.swiper-pagination',
            clickable: true,
        },
        navigation: {
            nextEl: '.swiper-button-next',
            prevEl: '.swiper-button-prev',
        },
    });
    
}


function displayList(elementId, heading, listData) {
    let ul = document.createElement('ul');
    ul.className = 'list';

    listData.forEach(item => {
        let li = document.createElement('li');
        li.textContent = item;
        ul.appendChild(li);
    });

    let container = document.getElementById(elementId);
    container.innerHTML = `<h3>${heading}</h3>`;
    container.appendChild(ul);
}
