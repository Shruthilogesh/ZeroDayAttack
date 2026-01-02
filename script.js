const slider = document.getElementById("threshold");
const tval = document.getElementById("tval");

slider.oninput = () => tval.innerText = slider.value;

function runDetection() {
    fetch(`/detect?threshold=${slider.value}`)
        .then(res => res.json())
        .then(data => {
            const tbody = document.getElementById("results");
            tbody.innerHTML = "";

            data.forEach(row => {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td>${row.index}</td>
                    <td>${row.score.toFixed(4)}</td>
                    <td class="${row.status}">${row.status}</td>
                `;
                tbody.appendChild(tr);
            });
        })
        .catch(err => alert(err));
}
