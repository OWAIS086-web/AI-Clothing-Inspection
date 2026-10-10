function addValue(setting) {
    const input = document.getElementById(setting);
    const value = input.value.trim();
    if (value) {
        const list = document.getElementById(setting + '_list');
        const box = document.createElement('div');
        box.className = 'value-box';
        box.innerHTML = value + ' <span class="remove" onclick="removeValue(this)">x</span>';
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
        box.innerHTML = fromValue + ' - ' + toValue + ' <span class="remove" onclick="removeValue(this)">x</span>';
        list.appendChild(box);
        fromInput.value = '';
        toInput.value = '';
    }
}

function removeValue(element) {
    element.parentElement.remove();
}

function saveSettings() {
    const settings = {
        production_mode: document.getElementById('production_mode').checked,
        ppta: document.getElementById('ppta').checked,
        searchmode: document.getElementById('searchmode').checked,
        column_names: getListValues('column_names_list'),
        sheet_name: getListValues('sheet_name_list'),
        fc_langs: getListValues('fc_langs_list'),
        care_langs: getListValues('care_langs_list'),
        excare_langs: getListValues('excare_langs_list'),
        COO_langs: getListValues('COO_langs_list'),
        care_symbols: getListValues('care_symbols_list'),
        order_market: getListValues('order_market_list'),
        fc_range: getRanges('fc_range_list'),
        excare_range: getRanges('excare_range_list'),
        care_range: getRanges('care_range_list'),
        symbol_range: getRanges('symbol_range_list')
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

function getValues(listId) {
    const list = document.getElementById(listId).children;
    const values = [];
    for (let i = 0; i < list.length; i++) {
        values.push(list[i].textContent.trim().replace(' x', ''));
    }
    return values.join('/');
}

function getRanges(listId) {
    const list = document.getElementById(listId).children;
    const ranges = [];
    for (let i = 0; i < list.length; i++) {
        const range = list[i].textContent.trim().replace(' x', '').split(' - ');
        ranges.push(range);
    }
    return ranges.flat(); // Flatten the array to avoid double brackets
}

function getListValues(listId) {
    const list = document.getElementById(listId).children;
    const values = [];
    for (let i = 0; i < list.length; i++) {
        values.push(list[i].textContent.trim().replace(' x', ''));
    }
    return values;
}

function loadDefaultSettings(settings) {
    for (const key in settings) {
        const value = settings[key];
        if (Array.isArray(value)) {
            const list = document.getElementById(key + '_list');
            value.forEach(item => {
                const box = document.createElement('div');
                box.className = 'value-box';
                box.innerHTML = item + ' <span class="remove" onclick="removeValue(this)">x</span>';
                list.appendChild(box);

                if (item.includes(' - ')) { // Check if the item is a range
                    const [fromValue, toValue] = item.split(' - ');
                    document.getElementById(key + '_from').value = fromValue;
                    document.getElementById(key + '_to').value = toValue;
                }
            });
        } else {
            const input = document.getElementById(key);
            if (input) {
                if (input.type === 'checkbox') {
                    input.checked = value;
                } else {
                    input.value = value;
                }
            }
        }
    }
}

// Example usage: Load default settings on page load
document.addEventListener('DOMContentLoaded', () => {
    fetch('/get_default_settings')
    .then(response => response.json())
    .then(data => {
        loadDefaultSettings(data);
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
