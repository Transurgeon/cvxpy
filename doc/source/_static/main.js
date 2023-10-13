$(document).ready(function() {
  // Create radio buttons and insert them before the table
  var radioButtons = `
    <h3>Curvature Filter</h3>
    <form id="filterForm">
      <label>
        <input type="radio" name="filter" value="all" checked> All
      </label>
      <label>
        <input type="radio" name="filter" value="convex"> Convex
      </label>
      <label>
        <input type="radio" name="filter" value="concave"> Concave
      </label>
    </form>
  `;
  var dropdownContent = document.querySelector(".sd-card-text");
  dropdownContent.innerHTML = radioButtons

  // Initialize DataTable
  var table = $('table.scalar').DataTable();

  // Add event listener to radio buttons for filtering
  $('input[name="filter"]').change(function() {
    var selectedValue = $(this).val();

    if (selectedValue === "all") {
      table.search('').columns().search('').draw();
    } else {
      // Apply the search filter to the specific column (index 4) in the table
      table.column(4).search(selectedValue).draw();
    }
  });
      // Create a container div
    var containerDiv = $('<div>');

    var convexTableTitle = `
      <div>
      <h3>Table of only Convex Atoms</h3>
      </div>
    `;
      // Create a new table after the original one
    var newTable = $('<table>').addClass('convex-view');

        // Get the filtered data from column 4 and extract the text from the <p> elements
    const convexData = $('table.scalar').DataTable().data().toArray().filter(row => $(row[4]).text().trim() === 'convex');

    // Add the filtered data to the new table
    newTable.DataTable({
      data: convexData,
      columns: [
        { title: "Function" },
        { title: "Meaning" },
        { title: "Domain" },
        { title: "Sign" },
        { title: "Curvature", visible: false },
        { title: "Monotonicity" },
      ]
    });

    // Append both convexTable and newTable to the container
    containerDiv.append(convexTableTitle);
    containerDiv.append(newTable);

    // Select the element containing the pagination controls (assuming it has a class like 'dataTables_paginate')
    var paginateControls = $('table.scalar').closest('.dataTables_wrapper').find('.dataTables_paginate');

    // Insert the container after the pagination controls
    paginateControls.after(containerDiv);
});
