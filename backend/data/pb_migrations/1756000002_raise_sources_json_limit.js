/// <reference path="../pb_data/types.d.ts" />
/**
 * Raise the sources_json text limit: the default 5000-char cap is smaller
 * than real exported message sources (observed up to ~27k chars). Also
 * applied to the init migration so fresh deployments match.
 */
migrate((app) => {
    const collection = app.findCollectionByNameOrId("messages");
    const field = collection.fields.getByName("sources_json");
    field.max = 100000;
    app.save(collection);
}, (app) => {
    const collection = app.findCollectionByNameOrId("messages");
    const field = collection.fields.getByName("sources_json");
    field.max = 5000;
    app.save(collection);
});
