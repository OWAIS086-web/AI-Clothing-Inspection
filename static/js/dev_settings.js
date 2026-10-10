function addRange(setting) {
    const fromInput = document.getElementById(setting + '_from');
    const toInput = document.getElementById(setting + '_to');
    const fromValue = fromInput.value.trim();
    const toValue = toInput.value.trim() || fromValue;

    if (fromValue === '') {
        alert('Please enter a valid "From" value.');
        return;
    }

    const list = document.getElementById(setting + '_list');
    const existingValues = new Set(Array.from(list.children).map(item => item.dataset.value));

    if (existingValues.has(fromValue) && existingValues.has(toValue)) {
        alert('These values already exist.');
        return;
    }

    const boxFrom = document.createElement('div');
    boxFrom.className = 'value-box';
    boxFrom.dataset.value = fromValue;
    boxFrom.innerHTML = `${fromValue} <span class="remove" onclick="removeValue(this)">x</span>`;
    list.appendChild(boxFrom);

    if (fromValue !== toValue) {
        const boxTo = document.createElement('div');
        boxTo.className = 'value-box';
        boxTo.dataset.value = toValue;
        boxTo.innerHTML = `${toValue} <span class="remove" onclick="removeValue(this)">x</span>`;
        list.appendChild(boxTo);
    }

    fromInput.value = '';
    toInput.value = '';
}

function removeValue(element) {
    element.parentElement.remove();
}

function clearContainer(setting) {
    const list = document.getElementById(setting + '_list');
    list.innerHTML = '';
}

function getRangeValues(listId) {
    const list = document.getElementById(listId).children;
    return Array.from(list).map(item => item.dataset.value);
}

function saveSettings() {
    const settings = {
        part_range: getRangeValues('part_range_list'),
        part_ext_range: getRangeValues('part_ext_range_list'),
        fiber_range_woppta: getRangeValues('fiber_range_woppta_list'),
        fiber_range_ppta: getRangeValues('fiber_range_ppta_list'),
        care_range: getRangeValues('care_range_list'),
        symbol_range: getRangeValues('symbol_range_list')
    };

    fetch('http://localhost:10002/save_settings', {
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

function loadDefaultSettings(settings) {
    for (const key in settings) {
        const value = settings[key];
        const list = document.getElementById(key + '_list');
        if (Array.isArray(value)) {
            const existingValues = new Set(Array.from(list.children).map(item => item.dataset.value));
            value.forEach(item => {
                if (!existingValues.has(item)) {
                    const box = document.createElement('div');
                    box.className = 'value-box';
                    box.dataset.value = item;
                    box.innerHTML = `${item} <span class="remove" onclick="removeValue(this)">x</span>`;
                    list.appendChild(box);
                }
            });
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    fetch('http://localhost:10002/get_default_settings')
    .then(response => response.json())
    .then(data => {
        loadDefaultSettings(data);
    })
    .catch(error => {
        console.error('Error:', error);
    });

    const rangeInputs = document.querySelectorAll('input[type="text"][id$="_from"], input[type="text"][id$="_to"]');
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
