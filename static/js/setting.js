function addValue(setting) {
    const input = document.getElementById(setting);
    const value = input.value.trim();
    if (value) {
        const list = document.getElementById(setting + '_list');
        const box = document.createElement('div');
        box.className = 'value-box';
        box.innerHTML = `${value} <span class="remove" onclick="removeValue(this)">x</span>`;
        list.appendChild(box);
        input.value = '';
    }
}

function addRange(setting) {
    const fromInput = document.getElementById(setting + '_from');
    const toInput = document.getElementById(setting + '_to');
    const fromValue = fromInput.value.trim();
    const toValue = toInput.value.trim() || fromValue; // If 'to' value is empty, use 'from' value
    if (fromValue) {
        const list = document.getElementById(setting + '_list');
        const box = document.createElement('div');
        box.className = 'value-box';
        box.innerHTML = `${fromValue} - ${toValue} <span class="remove" onclick="removeValue(this)">x</span>`;
        list.appendChild(box);
        fromInput.value = '';
        toInput.value = '';
    }
}

function removeValue(element) {
    element.parentElement.remove();
}

function clearContainer(setting) {
    const list = document.getElementById(setting + '_list');
    list.innerHTML = '';
}

function saveSettings() {
    const settings = {
        production_mode: document.getElementById('production_mode').checked,
        ppta: document.getElementById('ppta').checked,
        searchmode: document.getElementById('searchmode').checked,
        Size_Base_Sheet: document.getElementById('Size_Base_Sheet').checked,
        US_Size_Translation: document.getElementById('US_Size_Translation').checked,
        Adidas_Size_Translation: document.getElementById('Adidas_Size_Translation').checked,
        Body_Measurements_Languages: document.getElementById('Body_Measurements_Languages').checked,
        Size_Languages: getListValues('Size_Languages_list'),
        Size_Table_Range: getListValues('Size_Table_Range_list'),
        column_names: getListValues('column_names_list'),
        sheet_name: getListValues('sheet_name_list'),
        fc_langs: getListValues('fc_langs_list'),
        care_langs: getListValues('care_langs_list'),
        excare_langs: getListValues('excare_langs_list'),
        COO_langs: getListValues('COO_langs_list'),
        care_symbols: getListValues('care_symbols_list'),
        order_market: getListValues('order_market_list'),
        fc_range: getRangeValues('fc_range_list'),
        excare_range: getRangeValues('excare_range_list'),
        care_range: getRangeValues('care_range_list'),
        symbol_range: getRangeValues('symbol_range_list')
    };

    fetch('/save_settings', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(settings)
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function getListValues(listId) {
    const list = document.getElementById(listId).children;
    const values = [];
    for (let i = 0; i < list.length; i++) {
        values.push(list[i].textContent.trim().replace(' x', ''));
    }
    return values;
}

function getStringValues(listId) {
    const list = document.getElementById(listId).children;
    return Array.from(list).map(item => item.textContent.trim().replace(' x', '')).join('/');
}

function getRangeValues(listId) {
    const list = document.getElementById(listId).children;
    return Array.from(list).map(item => item.textContent.trim().replace(' x', '').replace(' - ', '/'));
}

function loadDefaultSettings(settings) {
    for (const key in settings) {
        const value = settings[key];
        if (Array.isArray(value)) {
            const list = document.getElementById(key + '_list');
            value.forEach(item => {
                const box = document.createElement('div');
                box.className = 'value-box';
                if (typeof item === 'object' && item.from && item.to) {
                    box.innerHTML = `${item.from} - ${item.to} <span class="remove" onclick="removeValue(this)">x</span>`;
                } else {
                    box.innerHTML = `${item} <span class="remove" onclick="removeValue(this)">x</span>`;
                }
                list.appendChild(box);
            });
        } else if (typeof value === 'boolean') {
            document.getElementById(key).checked = value;
        } else if (typeof value === 'string') {
            const values = value.split('/');
            const list = document.getElementById(key + '_list');
            values.forEach(item => {
                const box = document.createElement('div');
                box.className = 'value-box';
                box.innerHTML = `${item} <span class="remove" onclick="removeValue(this)">x</span>`;
                list.appendChild(box);
            });
        }
    }
}

function importJSON() {
    const fileInput = document.getElementById('json_file');
    const file = fileInput.files[0];
    if (file) {
        const formData = new FormData();
        formData.append('file', file);

        fetch('/upload_settings', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.message === "Import successful") {
                showPopup('importSuccessModal');
                document.getElementById('refreshPageButton').onclick = refreshPage;
            } else {
                alert(data.message);
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    } else {
        alert('Please select a JSON file to import.');
    }
}

function showPopup(modalId) {
    const modal = document.getElementById(modalId);
    const span = modal.querySelector('.close');
    modal.style.display = "block";
    span.onclick = function() {
        modal.style.display = "none";
    }
    window.onclick = function(event) {
        if (event.target == modal) {
            modal.style.display = "none";
        }
    }
}

function refreshPage() {
    window.location.reload();
}

document.addEventListener('DOMContentLoaded', () => {
    fetch('/get_default_settings')
    .then(response => response.json())
    .then(data => {
        loadDefaultSettings(data);
    })
    .catch(error => {
        console.error('Error:', error);
    });


    const singleValueInputs = document.querySelectorAll('input[type="text"]:not([id$="_from"]):not([id$="_to"])');
    singleValueInputs.forEach(input => {
        input.addEventListener('keypress', function (e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                addValue(this.id);
            }
        });
    });

    const rangeInputs = document.querySelectorAll('[id$="_from"], [id$="_to"]');
    rangeInputs.forEach(input => {
        input.addEventListener('keypress', function (e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                const setting = this.id.replace('_from', '').replace('_to', '');
                addRange(setting);
               
            }
        });
    });
});
